"""Test suite.

WHY TESTS IN A PORTFOLIO PROJECT
--------------------------------
Most portfolio repos have none. Having them is a cheap, visible differentiator:
it says you have worked on something that had to keep working.

But do not test everything. Test the things that fail *silently*, because those
are the ones that cost you a week:

  - Does a defect leak into the anomaly training split? (silently ruins AUROC)
  - Does freezing actually freeze? (silently destroys pretrained weights)
  - Does save/load reproduce identical scores? (silently serves a different
    model than the one you evaluated)
  - Does the cost threshold actually favour recall? (silently ships the wrong
    operating point)

A test that asserts `2 + 2 == 4` on your loss function is noise. These are not.

RUN:
    pytest -q
    pytest -q -m "not slow"     # skip the ones that build a dataset
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from visionqc.config import Config
from visionqc.data.datasets import (
    InspectionDataset, build_transforms, compute_class_weights, make_loader,
)
from visionqc.data.splits import build_splits
from visionqc.data.synthetic import generate_dataset
from visionqc.decision import Thresholds, Verdict, decide, normalise_anomaly
from visionqc.explain.gradcam import GradCAM
from visionqc.explain.overlay import denormalise, overlay_heatmap, side_by_side
from visionqc.metrics import (
    binary_metrics, cost_curve, localisation_iou, multiclass_report, pixel_auroc,
    safe_auroc, select_threshold_by_cost, select_threshold_by_recall,
)
from visionqc.models.autoencoder import ConvAutoencoder, gaussian_blur_map
from visionqc.models.classifier import DefectClassifier
from visionqc.models.padim import PaDiM
from visionqc.utils import count_parameters, get_device, set_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> Path:
    """A miniature dataset built once and shared by every test in the session."""
    root = tmp_path_factory.mktemp("data") / "tiny"
    generate_dataset(root, n_train_good=12, n_test_good=6, n_per_defect=3,
                     image_size=64, seed=3)
    return root


@pytest.fixture(scope="session")
def manifest(tiny_dataset) -> dict:
    return build_splits(tiny_dataset, holdout_frac=0.5, val_frac=0.25, seed=0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def test_generator_creates_mvtec_layout(tiny_dataset):
    assert (tiny_dataset / "train" / "good").is_dir()
    assert (tiny_dataset / "test" / "good").is_dir()
    for d in ("scratch", "dent", "contamination", "crack"):
        assert (tiny_dataset / "test" / d).is_dir()
        assert (tiny_dataset / "ground_truth" / d).is_dir()


def test_every_defect_image_has_a_mask(tiny_dataset):
    for d in ("scratch", "dent", "contamination", "crack"):
        imgs = sorted((tiny_dataset / "test" / d).glob("*.png"))
        masks = sorted((tiny_dataset / "ground_truth" / d).glob("*_mask.png"))
        assert len(imgs) == len(masks) > 0


def test_masks_are_binary_and_non_empty(tiny_dataset):
    from PIL import Image
    for d in ("scratch", "dent", "contamination", "crack"):
        for m in (tiny_dataset / "ground_truth" / d).glob("*_mask.png"):
            arr = np.array(Image.open(m))
            assert set(np.unique(arr)).issubset({0, 255}), f"{m} is not binary"
            assert arr.max() == 255, f"{m} marks no defect pixels at all"


def test_split_has_no_leak(manifest):
    """The headline invariant: test images appear in no training split."""
    test_paths = {r["path"] for r in manifest["splits"]["test"]}
    for name in ("anomaly_fit", "sup_train", "sup_val"):
        other = {r["path"] for r in manifest["splits"][name]}
        assert not (test_paths & other), f"leak between test and {name}"


def test_anomaly_fit_contains_no_defects(manifest):
    """If this ever fails, the unsupervised path is silently meaningless."""
    assert all(r["is_defect"] == 0 for r in manifest["splits"]["anomaly_fit"])


def test_split_is_deterministic(tiny_dataset):
    a = build_splits(tiny_dataset, 0.5, 0.25, seed=11)
    b = build_splits(tiny_dataset, 0.5, 0.25, seed=11)
    assert [r["path"] for r in a["splits"]["test"]] == \
           [r["path"] for r in b["splits"]["test"]]


def test_different_seed_gives_different_split(tiny_dataset):
    a = build_splits(tiny_dataset, 0.5, 0.25, seed=1)
    b = build_splits(tiny_dataset, 0.5, 0.25, seed=2)
    assert [r["path"] for r in a["splits"]["test"]] != \
           [r["path"] for r in b["splits"]["test"]]


def test_class_zero_is_good(manifest):
    """The engine reads P(defect) as 1 - P(class 0). If 'good' is not index 0,
    every binary probability is wrong."""
    assert manifest["classes"][0] == "good"
    assert manifest["class_to_idx"]["good"] == 0


def test_test_split_contains_both_classes(manifest):
    labels = {r["is_defect"] for r in manifest["splits"]["test"]}
    assert labels == {0, 1}, "AUROC is undefined without both classes in test"


def test_dataset_item_shapes(manifest):
    ds = InspectionDataset(manifest["splits"]["test"], manifest["class_to_idx"],
                           image_size=64, load_masks=True)
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["image"].dtype == torch.float32
    assert item["mask"].shape == (64, 64)
    assert set(np.unique(item["mask"].numpy())).issubset({0, 1})


def test_eval_transform_is_deterministic(manifest):
    ds = InspectionDataset(manifest["splits"]["test"], manifest["class_to_idx"],
                           image_size=64, train=False)
    assert torch.equal(ds[0]["image"], ds[0]["image"])


def test_train_transform_actually_augments(manifest):
    set_seed(0)
    ds = InspectionDataset(manifest["splits"]["sup_train"], manifest["class_to_idx"],
                           image_size=64, train=True)
    assert not torch.equal(ds[0]["image"], ds[0]["image"])


def test_class_weights_favour_rare_classes(manifest):
    w = compute_class_weights(manifest["splits"]["sup_train"], manifest["class_to_idx"])
    assert w.shape[0] == len(manifest["classes"])
    # 'good' is the most common class, so it must get the smallest weight.
    assert w[0] == w.min()
    assert torch.isfinite(w).all()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def test_classifier_output_shape():
    m = DefectClassifier(5, "resnet18", pretrained=False)
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 5)


def test_freeze_actually_freezes():
    """The classic silent bug: you think the backbone is frozen and it is not."""
    m = DefectClassifier(3, "resnet18", pretrained=False)
    _, before = count_parameters(m)
    m.freeze_backbone()
    _, after = count_parameters(m)
    assert after < before
    assert all(not p.requires_grad for p in m.net.parameters())
    # BatchNorm must also stop updating its running statistics.
    assert not m.net.training
    m.unfreeze_backbone()
    assert count_parameters(m)[1] == before


def test_frozen_backbone_receives_no_gradients():
    m = DefectClassifier(3, "resnet18", pretrained=False)
    m.freeze_backbone()
    m(torch.randn(2, 3, 64, 64)).sum().backward()
    assert all(p.grad is None for p in m.net.parameters())
    assert any(p.grad is not None for p in m.head.parameters())


def test_param_groups_use_different_lrs():
    m = DefectClassifier(3, "resnet18", pretrained=False)
    groups = m.param_groups(lr_head=1e-3, lr_backbone=1e-4)
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3


def test_autoencoder_reconstructs_input_shape():
    ae = ConvAutoencoder(base_channels=8, latent_channels=16)
    x = torch.randn(2, 3, 64, 64)
    assert ae(x).shape == x.shape
    assert ae.encoder(x).shape == (2, 16, 4, 4)   # 16x compression in space
    assert ae.anomaly_map(x).shape == (2, 64, 64)


def test_gaussian_blur_preserves_shape_and_mass():
    m = torch.zeros(1, 32, 32)
    m[0, 16, 16] = 100.0
    b = gaussian_blur_map(m, sigma=2.0)
    assert b.shape == m.shape
    assert b.max() < m.max()                 # the spike is spread out
    assert b[0, 16, 16] > b[0, 0, 0]         # but still centred where it was
    assert torch.equal(gaussian_blur_map(m, 0.0), m)   # sigma=0 is a no-op


@pytest.mark.slow
def test_padim_fit_and_score(manifest):
    ds = InspectionDataset(manifest["splits"]["anomaly_fit"],
                           manifest["class_to_idx"], image_size=64)
    loader = make_loader(ds, 4, False, num_workers=0)
    p = PaDiM(n_components=16, embed_grid=8, pretrained=False).eval()
    p.fit(loader, torch.device("cpu"))
    assert p.is_fitted
    x = torch.randn(2, 3, 64, 64)
    assert p.anomaly_map(x).shape == (2, 64, 64)
    assert p.image_score(x).shape == (2,)
    assert torch.isfinite(p.image_score(x)).all()


@pytest.mark.slow
def test_padim_scores_noise_higher_than_normal(manifest):
    """Sanity: pure noise must look more anomalous than a real normal part."""
    ds = InspectionDataset(manifest["splits"]["anomaly_fit"],
                           manifest["class_to_idx"], image_size=64)
    loader = make_loader(ds, 4, False, num_workers=0)
    p = PaDiM(n_components=16, embed_grid=8, pretrained=False).eval()
    p.fit(loader, torch.device("cpu"))
    normal = torch.stack([ds[i]["image"] for i in range(4)])
    noise = torch.randn_like(normal)
    assert p.image_score(noise).mean() > p.image_score(normal).mean()


@pytest.mark.slow
def test_padim_roundtrip_is_bit_identical(manifest, tmp_path):
    """Guards against serving a different model than the one you evaluated."""
    ds = InspectionDataset(manifest["splits"]["anomaly_fit"],
                           manifest["class_to_idx"], image_size=64)
    loader = make_loader(ds, 4, False, num_workers=0)
    p = PaDiM(n_components=16, embed_grid=8, pretrained=False).eval()
    p.fit(loader, torch.device("cpu"))
    x = torch.stack([ds[i]["image"] for i in range(3)])
    before = p.image_score(x)
    path = tmp_path / "padim.pt"
    p.save(path)
    after = PaDiM.load(path).image_score(x)
    assert torch.allclose(before, after, atol=1e-4)


def test_padim_rejects_scoring_before_fit():
    p = PaDiM(pretrained=False)
    with pytest.raises(RuntimeError, match="fit"):
        p.anomaly_map(torch.randn(1, 3, 64, 64))


def test_padim_rejects_too_few_images():
    p = PaDiM(n_components=8, embed_grid=4, pretrained=False)
    loader = [{"image": torch.randn(1, 3, 64, 64)}]
    with pytest.raises(ValueError, match=">= 2"):
        p.fit(loader, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def test_gradcam_shape_and_range():
    m = DefectClassifier(4, "resnet18", pretrained=False)
    with GradCAM(m, m.target_layer()) as cam:
        heat, idx = cam(torch.randn(2, 3, 64, 64))
    assert heat.shape == (2, 64, 64)
    assert 0.0 <= float(heat.min()) and float(heat.max()) <= 1.0
    assert idx.shape == (2,)


def test_gradcam_removes_hooks():
    """A leaked hook keeps tensors alive and leaks memory across API requests."""
    m = DefectClassifier(3, "resnet18", pretrained=False)
    with GradCAM(m, m.target_layer()) as cam:
        cam(torch.randn(1, 3, 64, 64))
    assert cam._handles == []


def test_gradcam_target_class_changes_map():
    m = DefectClassifier(4, "resnet18", pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    with GradCAM(m, m.target_layer()) as cam:
        a, _ = cam(x, class_idx=0)
    with GradCAM(m, m.target_layer()) as cam:
        b, _ = cam(x, class_idx=3)
    assert not torch.allclose(a, b)


def test_gradcam_requires_context_manager():
    m = DefectClassifier(3, "resnet18", pretrained=False)
    cam = GradCAM(m, m.target_layer())
    with pytest.raises(RuntimeError, match="with"):
        cam(torch.randn(1, 3, 64, 64))


def test_overlay_shapes():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    heat = np.random.rand(64, 64)
    out = overlay_heatmap(img, heat)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert side_by_side([img, out]).shape[1] > img.shape[1]


def test_overlay_resizes_mismatched_map():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert overlay_heatmap(img, np.random.rand(8, 8)).shape == (64, 64, 3)


def test_denormalise_roundtrip():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    original = torch.rand(3, 32, 32)
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    out = denormalise((original - m) / s, mean, std)
    assert np.abs(out.astype(float) / 255 - original.permute(1, 2, 0).numpy()).max() < 0.01


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_binary_metrics_on_a_perfect_model():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    m = binary_metrics(y, s, 0.5)
    assert m.recall == 1.0 and m.precision == 1.0 and m.auroc == 1.0
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 0, 2, 0)


def test_binary_metrics_on_an_all_pass_model():
    """The failure mode we must never ship: high accuracy, zero recall."""
    y = [0] * 98 + [1] * 2
    s = [0.0] * 100
    m = binary_metrics(y, s, 0.5)
    assert m.accuracy == 0.98
    assert m.recall == 0.0        # accuracy looks great, model is useless


def test_safe_auroc_handles_single_class():
    assert np.isnan(safe_auroc(np.zeros(5), np.arange(5)))


def test_cost_threshold_favours_recall_when_fn_is_expensive():
    """The central claim of the decision layer, asserted."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 20)
    s = np.concatenate([rng.normal(0.3, 0.15, 100), rng.normal(0.6, 0.15, 20)])
    expensive_fn = select_threshold_by_cost(y, s, cost_fn=50, cost_fp=1)
    balanced = select_threshold_by_cost(y, s, cost_fn=1, cost_fp=1)
    assert expensive_fn.threshold <= balanced.threshold
    assert expensive_fn.recall >= balanced.recall


def test_cost_threshold_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        select_threshold_by_cost([0, 0, 0], [0.1, 0.2, 0.3])


def test_recall_target_is_met_when_achievable():
    y = [0] * 20 + [1] * 20
    s = list(np.linspace(0, 0.5, 20)) + list(np.linspace(0.5, 1.0, 20))
    c = select_threshold_by_recall(y, s, target_recall=0.9)
    assert c.recall >= 0.9


def test_unreachable_recall_degrades_gracefully():
    c = select_threshold_by_recall([0, 1], [0.5, 0.5], target_recall=1.01)
    assert c.strategy == "recall_target_unreachable"
    assert "improve the model" in c.rationale


def test_cost_curve_minimum_matches_chosen_threshold():
    rng = np.random.default_rng(1)
    y = np.array([0] * 60 + [1] * 15)
    s = np.concatenate([rng.normal(0.3, 0.1, 60), rng.normal(0.7, 0.1, 15)])
    chosen = select_threshold_by_cost(y, s, 10, 1)
    curve = cost_curve(y, s, 10, 1)
    assert chosen.expected_cost <= min(curve["cost"]) + 1e-9


def test_multiclass_macro_recall_punishes_an_ignored_class():
    classes = ["good", "a", "b"]
    # Class 'b' is never predicted -> macro recall must drop well below accuracy.
    y = [0, 0, 1, 2]
    p = [0, 0, 1, 1]
    rep = multiclass_report(y, p, classes)
    assert rep["per_class"]["b"]["recall"] == 0.0
    assert rep["macro_recall"] < rep["accuracy"]


def test_pixel_auroc_perfect_and_random():
    mask = np.zeros((2, 16, 16), dtype=np.uint8)
    mask[:, 4:8, 4:8] = 1
    assert pixel_auroc(mask, mask.astype(float)) == 1.0
    assert np.isnan(pixel_auroc(np.zeros((2, 8, 8), np.uint8), np.random.rand(2, 8, 8)))


def test_localisation_iou_bounds():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    assert localisation_iou(mask, mask.astype(float), quantile=0.9) > 0.5
    assert np.isnan(localisation_iou(np.zeros((8, 8), np.uint8), np.random.rand(8, 8)))


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------
def test_pass_when_both_signals_are_quiet():
    t = Thresholds(classifier=0.5, anomaly=10.0)
    d = decide(t, defect_probability=0.1, predicted_class="good", anomaly_score=2.0)
    assert d.verdict is Verdict.PASS


def test_fail_classified_when_classifier_names_a_defect():
    t = Thresholds(classifier=0.5, anomaly=10.0)
    d = decide(t, defect_probability=0.9, predicted_class="scratch", anomaly_score=2.0)
    assert d.verdict is Verdict.FAIL_CLASSIFIED
    assert "scratch" in " ".join(d.reasons)


def test_fail_anomaly_for_an_unseen_defect_type():
    """The scenario that justifies the whole dual-path design."""
    t = Thresholds(classifier=0.5, anomaly=10.0)
    d = decide(t, defect_probability=0.1, predicted_class="good", anomaly_score=50.0)
    assert d.verdict is Verdict.FAIL_ANOMALY


def test_fusion_catches_what_the_classifier_misses():
    t = Thresholds(classifier=0.5, anomaly=10.0)
    clf_only = decide(t, 0.1, "good", None, 50.0, primary="classifier")
    fused = decide(t, 0.1, "good", None, 50.0, primary="fusion")
    assert clf_only.verdict is Verdict.PASS
    assert fused.verdict is not Verdict.PASS


def test_review_when_classifier_flags_but_cannot_name():
    t = Thresholds(classifier=0.5, anomaly=10.0)
    d = decide(t, defect_probability=0.9, predicted_class="good", anomaly_score=2.0)
    assert d.verdict is Verdict.REVIEW


def test_missing_signals_never_silently_pass():
    """A model that failed to load must not turn into a stream of PASS verdicts."""
    d = decide(Thresholds(), defect_probability=None, anomaly_score=None)
    assert d.verdict is Verdict.REVIEW


def test_uncalibrated_threshold_is_ignored_not_defaulted():
    t = Thresholds(classifier=None, anomaly=10.0)
    d = decide(t, defect_probability=0.99, predicted_class="scratch", anomaly_score=1.0)
    assert d.classifier_flag is False      # no threshold -> no vote
    assert d.verdict is Verdict.PASS


def test_unknown_primary_raises():
    with pytest.raises(ValueError, match="Unknown decision.primary"):
        decide(Thresholds(anomaly=1.0), anomaly_score=2.0, primary="nonsense")


def test_normalise_anomaly_is_relative_to_good_parts():
    assert normalise_anomaly(20.0, 10.0) == 2.0
    assert normalise_anomaly(20.0, None) is None
    assert normalise_anomaly(20.0, 0.0) is None


def test_verdict_is_json_serialisable():
    d = decide(Thresholds(anomaly=1.0), anomaly_score=5.0)
    assert json.loads(json.dumps(d.to_dict()))["verdict"] == "FAIL_ANOMALY"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.run_name = "unit"
    p = tmp_path / "c.yaml"
    cfg.save(p)
    assert Config.from_yaml(p).run_name == "unit"


def test_config_rejects_typos():
    with pytest.raises(ValueError, match="Unknown config key"):
        Config.from_dict({"seeed": 1})
    with pytest.raises(ValueError, match="Unknown keys for"):
        Config.from_dict({"padim": {"n_componentz": 5}})


def test_config_coerces_yaml_lists_to_tuples():
    cfg = Config.from_dict({"padim": {"layers": ["layer1", "layer2"]}})
    assert cfg.padim.layers == ("layer1", "layer2")


def test_get_device_respects_explicit_choice():
    assert get_device("cpu").type == "cpu"


def test_set_seed_makes_runs_reproducible():
    set_seed(3); a = torch.randn(4)
    set_seed(3); b = torch.randn(4)
    assert torch.equal(a, b)
