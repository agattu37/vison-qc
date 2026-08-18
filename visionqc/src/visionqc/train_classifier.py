"""Train the supervised defect classifier.

RUN:
    python -m visionqc.train_classifier --config configs/synthetic.yaml

WHAT THIS SCRIPT DOES, AND WHY IN THIS ORDER
--------------------------------------------
1. Load the split manifest (built once, reused by every script -> all models
   see the same data partition, so their numbers are comparable).
2. Phase 1: freeze the backbone, train only the new head.
3. Phase 2: unfreeze, fine-tune everything with discriminative learning rates.
4. Track validation macro-recall each epoch; keep the best weights.
5. Save the checkpoint, the training history, and the exact config used.

WHY MACRO-RECALL IS THE MODEL-SELECTION METRIC
----------------------------------------------
Not validation loss, and definitely not validation accuracy.

Accuracy is dominated by the 'good' class -- a model that never predicts
'crack' can still score well. Loss is a proxy that can improve while the metric
you care about gets worse (it rewards confidence on easy examples).

Macro-recall averages recall over classes with equal weight, so failing
completely on one rare defect type is heavily penalised. That is precisely the
failure we must not ship. Selecting on the metric you actually care about is a
small discipline that pays off constantly.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data.datasets import InspectionDataset, compute_class_weights, make_loader
from .metrics import multiclass_report
from .models.classifier import build_classifier
from .utils import count_parameters, get_device, get_logger, load_json, save_json, set_seed

logger = get_logger("visionqc.train_clf")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """One pass over a loader. Returns (mean_loss, y_true, y_pred).

    Sharing this between train and eval keeps the two paths from drifting apart
    -- a classic source of "why is my val accuracy weirdly high" bugs where the
    two code paths preprocess differently.
    """
    model.train(train)
    # If the backbone is frozen we keep it in eval() so its BatchNorm running
    # statistics stay fixed. model.train(True) would have just undone that.
    if train and not any(p.requires_grad for p in model.net.parameters()):
        model.net.eval()

    total_loss, n = 0.0, 0
    ys, ps = [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping: cheap insurance. With a tiny defect class and a
            # weighted loss, one badly-scaled batch can produce a huge gradient
            # that undoes several epochs of progress.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        ys.append(y.detach().cpu().numpy())
        ps.append(logits.detach().argmax(1).cpu().numpy())

    return total_loss / max(n, 1), np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the VisionQC defect classifier.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default=None, help="defaults to <run_dir>/splits.json")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    splits_path = Path(args.splits) if args.splits else run_dir / "splits.json"
    manifest = load_json(splits_path)
    classes = manifest["classes"]
    c2i = manifest["class_to_idx"]

    logger.info("Device: %s | classes: %s", device, classes)

    d = cfg.data
    train_ds = InspectionDataset(
        manifest["splits"]["sup_train"], c2i, d.image_size, d.mean, d.std, train=True
    )
    val_ds = InspectionDataset(
        manifest["splits"]["sup_val"], c2i, d.image_size, d.mean, d.std, train=False
    )
    train_loader = make_loader(train_ds, cfg.classifier.batch_size, True, d.num_workers)
    val_loader = make_loader(val_ds, cfg.classifier.batch_size, False, d.num_workers)
    logger.info("Train: %d images | Val: %d images", len(train_ds), len(val_ds))

    model = build_classifier(cfg.classifier, num_classes=len(classes)).to(device)
    total, trainable = count_parameters(model)
    logger.info("Model: %s | %.1fM params", cfg.classifier.backbone, total / 1e6)

    # Class weighting computed on the TRAIN split only. Using the full dataset
    # would leak information about the validation class balance into training.
    weights = None
    if cfg.classifier.use_class_weights:
        weights = compute_class_weights(manifest["splits"]["sup_train"], c2i).to(device)
        logger.info(
            "Class weights: %s",
            {c: round(float(weights[i]), 2) for c, i in sorted(c2i.items(), key=lambda kv: kv[1])},
        )

    # Label smoothing: stops the model driving logits to infinity on a training
    # set this small, which is a form of overfitting that also ruins the
    # calibration of the probabilities we later threshold.
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=cfg.classifier.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.classifier.lr_head, cfg.classifier.lr_backbone),
        weight_decay=cfg.classifier.weight_decay,
    )
    # Cosine annealing: a high LR early to explore, decaying to a low LR that
    # settles into a minimum. Well-behaved and needs no tuning, which matters
    # when compute is limited and you cannot afford an LR sweep.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.classifier.epochs)
    )

    model.freeze_backbone()
    logger.info("Phase 1: backbone FROZEN, %d trainable params", count_parameters(model)[1])

    best_score, best_state, best_epoch, patience = -1.0, None, -1, 0
    history: list[dict] = []

    for epoch in range(cfg.classifier.epochs):
        if epoch == cfg.classifier.freeze_epochs:
            model.unfreeze_backbone()
            logger.info(
                "Phase 2: backbone UNFROZEN, %d trainable params",
                count_parameters(model)[1],
            )

        tr_loss, tr_y, tr_p = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        va_loss, va_y, va_p = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        tr_rep = multiclass_report(tr_y, tr_p, classes)
        va_rep = multiclass_report(va_y, va_p, classes)
        score = va_rep["macro_recall"]

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "train_macro_recall": tr_rep["macro_recall"],
            "val_macro_recall": score, "val_macro_f1": va_rep["macro_f1"],
            "val_accuracy": va_rep["accuracy"],
            "lr": optimizer.param_groups[-1]["lr"],
        })
        logger.info(
            "epoch %02d | train loss %.4f rec %.3f | val loss %.4f rec %.3f f1 %.3f",
            epoch, tr_loss, tr_rep["macro_recall"], va_loss, score, va_rep["macro_f1"],
        )

        if score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            # deepcopy to CPU: keeping a GPU copy of the best weights around
            # wastes VRAM you may need for the next forward pass.
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
        else:
            patience += 1
            if patience >= cfg.classifier.early_stop_patience:
                logger.info("Early stop at epoch %d (no gain for %d epochs)",
                            epoch, patience)
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint -- did the loader yield data?")

    model.load_state_dict(best_state)
    ckpt_path = run_dir / "classifier.pt"
    torch.save(
        {
            "state_dict": best_state,
            "classes": classes,
            "backbone": cfg.classifier.backbone,
            "image_size": d.image_size,
            "mean": list(d.mean), "std": list(d.std),
            "best_epoch": best_epoch, "best_val_macro_recall": best_score,
        },
        ckpt_path,
    )
    save_json(history, run_dir / "classifier_history.json")
    cfg.save(run_dir / "config_used.yaml")

    logger.info("Best epoch %d | val macro-recall %.4f", best_epoch, best_score)
    logger.info("Saved -> %s", ckpt_path)


if __name__ == "__main__":
    main()
