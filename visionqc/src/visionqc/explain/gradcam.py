"""Grad-CAM, implemented from scratch with PyTorch hooks.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).

WHY EXPLAINABILITY IS NOT OPTIONAL HERE
---------------------------------------
A QC operator will not scrap a part because a black box said 0.93. They need to
see *where*. Beyond usability, the heatmap is your debugging tool: it is how you
discover that your model is keyed on the conveyor belt edge or a timestamp
overlay rather than the part. That failure mode -- a model that is accurate on
your test set for entirely the wrong reason -- is common, and Grad-CAM is how
you catch it before it ships.

HOW IT WORKS, WITHOUT THE MATHS
-------------------------------
Take the last convolutional layer. It outputs, say, 512 feature maps of 8x8.
Each map is a detector for some pattern, and it still knows *where* that pattern
occurred, because it has not been flattened yet.

We want to know which of those 512 detectors mattered for the predicted class.
Backpropagation answers exactly that: the gradient of the class score with
respect to each map tells us how much nudging that map would change the score.
Average each map's gradient into a single number and you have its importance
weight.

Then: weighted sum of the maps, using those weights. Apply ReLU (we only want
evidence *for* the class, not against it). Upsample the 8x8 result to the input
size. That is the heatmap.

    weight_k    = mean over spatial positions of  d(score_c) / d(A_k)
    heatmap     = ReLU( sum over k of  weight_k * A_k )

WHY NOT `pip install grad-cam`
------------------------------
The library is good and we mention it in the docs. We hand-roll here for two
practical reasons: it drops an OpenCV dependency (OpenCV 5 is a recent major
release, and pinning it in a slim container is friction we do not need), and
being able to explain hooks is worth more in an interview than the 40 lines we
saved.

KNOWN LIMITATION -- say this before an interviewer says it to you
-----------------------------------------------------------------
Grad-CAM's resolution is the resolution of the layer it hooks. At 256x256 input,
ResNet18's layer4 is 8x8. Upsampling gives you a blurry blob, which is fine for
"the defect is in this region" and useless for "the scratch is exactly these
pixels". For crisp localisation, the PaDiM map is the better output -- it works
at a 32x32 grid and is not tied to a predicted class. Knowing which tool answers
which question is the point of building both.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM maps for a classifier.

    Usage as a context manager guarantees the hooks are removed even if
    something raises. Leaked hooks are nasty: they silently keep tensors alive,
    leak memory across requests in a long-running API, and can change behaviour
    of later forward passes.

        with GradCAM(model, model.target_layer()) as cam:
            heatmap, cls_idx = cam(images)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list[Any] = []

    def __enter__(self) -> "GradCAM":
        self._register()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()

    def _register(self) -> None:
        def forward_hook(_module, _inp, output):
            # Keep the activation for the weighted sum later.
            self.activations = output
            # Registering a hook on the *tensor* is the modern, reliable way to
            # capture its gradient. `register_full_backward_hook` on the module
            # also works but is fiddlier with modules that have multiple outputs.
            if output.requires_grad:
                output.register_hook(self._save_grad)

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))

    def _save_grad(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.activations = None
        self.gradients = None

    def __call__(
        self,
        images: torch.Tensor,
        class_idx: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (heatmaps (B,H,W) in [0,1], class indices (B,)).

        `class_idx=None` explains the model's own top prediction, which is what
        you want when auditing. Pass an explicit index to ask the
        counterfactual question "what would have made you say *scratch*?".
        """
        if not self._handles:
            raise RuntimeError("Use GradCAM inside a `with` block so hooks are live.")

        was_training = self.model.training
        self.model.eval()

        # Grad-CAM needs a backward pass, so we must NOT be under no_grad, even
        # at inference time. This is the mistake that makes people think
        # Grad-CAM "returns all zeros".
        with torch.enable_grad():
            images = images.clone().requires_grad_(False)
            logits = self.model(images)

            if class_idx is None:
                target = logits.argmax(dim=1)
            elif isinstance(class_idx, int):
                target = torch.full(
                    (logits.shape[0],), class_idx, dtype=torch.long, device=logits.device
                )
            else:
                target = class_idx.to(logits.device)

            # Sum the target logits across the batch. Because each sample's logit
            # depends only on its own activations, one backward pass gives every
            # sample the correct per-sample gradient -- no Python loop needed.
            score = logits.gather(1, target.view(-1, 1)).sum()

            self.model.zero_grad(set_to_none=True)
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "No activations/gradients captured. Is the target layer actually "
                "part of the forward pass?"
            )

        acts = self.activations.detach()          # (B, C, h, w)
        grads = self.gradients                    # (B, C, h, w)

        weights = grads.mean(dim=(2, 3), keepdim=True)        # (B, C, 1, 1)
        cam = (weights * acts).sum(dim=1)                      # (B, h, w)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1), size=images.shape[-2:],
            mode="bilinear", align_corners=False,
        ).squeeze(1)

        cam = _normalise_per_sample(cam)

        if was_training:
            self.model.train()
        return cam.detach(), target.detach()


def _normalise_per_sample(cam: torch.Tensor) -> torch.Tensor:
    """Scale each heatmap to [0, 1] independently.

    Per-sample, not per-batch: the raw magnitudes are not comparable across
    images anyway, and normalising over the batch would make one hot image wash
    out every other map in the same batch -- a genuinely confusing artefact when
    you are eyeballing results in a grid.
    """
    b = cam.shape[0]
    flat = cam.view(b, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    # eps guards the all-zero case (a map with no positive evidence at all).
    return ((flat - lo) / (hi - lo + 1e-8)).view_as(cam)
