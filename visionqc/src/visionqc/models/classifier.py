"""Supervised defect classifier via transfer learning.

WHY TRANSFER LEARNING AT ALL
----------------------------
We have on the order of tens of labelled defects per class. Training a CNN from
random initialisation on that is hopeless — it would memorise the training set
before learning anything general.

A ResNet18 pretrained on ImageNet has already learned, in its early layers, the
things that are expensive to learn: edges, corners, textures, gradients. A
scratch on metal is an edge. A dent is a shading gradient. Those detectors
transfer almost perfectly, even though ImageNet contains no machined parts. We
only need to teach the last layers what to *do* with those features.

WHY STAGED UNFREEZING (the `freeze_epochs` idea)
------------------------------------------------
On epoch 1 the classifier head is random, so it produces large, meaningless
gradients. If the backbone is trainable at that moment, those garbage gradients
flow backwards and damage the pretrained weights you paid nothing for. This is
sometimes called "catastrophic forgetting" of the pretrained features.

So we do it in two phases:
  Phase 1 (epochs 0..freeze_epochs-1): backbone frozen, train the head only.
      The head learns a sane mapping from good features to our classes.
  Phase 2 (remaining epochs): unfreeze, and fine-tune the whole network with a
      *smaller* learning rate on the backbone than on the head.

That second detail — discriminative learning rates — matters. The backbone needs
gentle nudges; the head needs real learning. One global LR cannot do both.

WHY RESNET18 AND NOT SOMETHING BIGGER
--------------------------------------
Three reasons, all defensible in an interview:
  1. ~11M parameters fits comfortably in free-tier GPU memory and trains in
     minutes, so you can run many experiments.
  2. Its residual stages give clean, well-separated feature maps at 4 scales,
     which the PaDiM path reuses directly. One backbone, two models.
  3. It runs a single 256x256 image on CPU in well under our 2s latency budget.
     A ViT or EfficientNet-B4 would not, on free hosting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, EfficientNet_B0_Weights,
    resnet18, resnet34, efficientnet_b0,
)

# Registry so the config string maps to a real constructor without eval().
_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
}


class DefectClassifier(nn.Module):
    """Pretrained backbone + a small custom head.

    Output is logits over N classes, where class 0 is 'good' by construction of
    the manifest. That lets us read a binary defect probability straight off the
    multiclass softmax as `1 - P(good)`, so one model gives us both the
    "is it defective?" decision and the "what kind of defect?" answer.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Available: {sorted(_BACKBONES)}"
            )
        ctor, weights = _BACKBONES[backbone]
        # NOTE the modern API: weights=<Enum>, not the deprecated pretrained=True.
        # Passing pretrained=True raises in current torchvision.
        self.backbone_name = backbone
        self.net = ctor(weights=weights if pretrained else None)

        if backbone.startswith("resnet"):
            in_features = self.net.fc.in_features
            self.net.fc = nn.Identity()      # strip the 1000-class ImageNet head
        else:                                 # efficientnet
            in_features = self.net.classifier[1].in_features
            self.net.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.num_classes = num_classes
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))

    # ---- freezing control -------------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.net.parameters():
            p.requires_grad = False
        # BatchNorm keeps updating running statistics even when its weights are
        # frozen, which shifts the features under the head. Putting the backbone
        # in eval() mode stops that. This is the classic subtle freezing bug.
        self.net.eval()

    def unfreeze_backbone(self) -> None:
        for p in self.net.parameters():
            p.requires_grad = True
        self.net.train()

    def param_groups(self, lr_head: float, lr_backbone: float) -> list[dict]:
        """Discriminative learning rates: gentle on pretrained, bold on new."""
        return [
            {"params": self.net.parameters(), "lr": lr_backbone},
            {"params": self.head.parameters(), "lr": lr_head},
        ]

    def target_layer(self) -> nn.Module:
        """The layer Grad-CAM hooks into.

        We want the *last* convolutional stage: deep enough to be semantic
        ("this region looks defective"), but still spatial (8x8 for a 256px
        input) so it can be upsampled into a heatmap. The layer after it is
        global average pooling, which destroys all spatial information — hook
        there and you get a uniform, useless map.
        """
        if self.backbone_name.startswith("resnet"):
            return self.net.layer4[-1]
        return self.net.features[-1]


def build_classifier(cfg, num_classes: int) -> DefectClassifier:
    return DefectClassifier(
        num_classes=num_classes,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
    )
