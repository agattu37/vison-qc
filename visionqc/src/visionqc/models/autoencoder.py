"""Unsupervised baseline: convolutional autoencoder.

THE CORE IDEA, IN ONE PARAGRAPH
-------------------------------
Train a network to compress an image into a small code and rebuild it, using
*only* normal parts. It becomes very good at rebuilding normal parts and only
normal parts, because that is all it has ever seen. Show it a scratched part and
it rebuilds the part but not the scratch — it has no vocabulary for scratches.
Subtract the reconstruction from the input and the leftover error lights up
exactly where the anomaly is. The error map is your heatmap; its aggregate is
your anomaly score.

WHY THE BOTTLENECK MUST BE TIGHT
--------------------------------
If the latent code is large enough, the autoencoder learns the identity function
— it copies input to output, reconstructs defects perfectly, and detects
nothing. The bottleneck is what forces it to learn a *compressed model of
normality* rather than a copy. This is the number one reason beginner
autoencoders fail at anomaly detection, and "how did you size the bottleneck?"
is a fair interview question. Our default: 256x256x3 = 196,608 values in,
64x16x16 = 16,384 in the code. An 12x compression.

WHY WE STILL EXPECT IT TO UNDERPERFORM PADIM
--------------------------------------------
Be upfront about this — it is the honest finding that makes the project
credible. Autoencoders have two known weaknesses:
  1. They generalise too well. A sufficiently smooth defect gets reconstructed
     anyway, especially low-contrast ones like our 'crack' class.
  2. Per-pixel L2 error is dominated by high-frequency texture. A perfectly
     normal but slightly misaligned edge produces more error than a genuine but
     subtle defect.
PaDiM sidesteps both by comparing *pretrained features* rather than raw pixels.
We build the autoencoder first anyway, because a baseline you can beat is how
you demonstrate that your upgrade was actually worth it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(cin: int, cout: int, stride: int = 2) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU. BN keeps activations well-scaled so a deep
    stack trains without careful manual initialisation."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def _upconv_block(cin: int, cout: int, final: bool = False) -> nn.Sequential:
    """Upsample + Conv rather than ConvTranspose2d.

    ConvTranspose2d produces checkerboard artefacts when kernel size is not
    divisible by stride. In an anomaly detector those artefacts become fake
    "errors" scattered across the whole image, which is exactly the signal we
    are trying to read. Nearest-neighbour upsampling followed by a normal conv
    avoids the problem entirely at negligible cost.
    """
    layers: list[nn.Module] = [
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=not final),
    ]
    if not final:
        layers += [nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class ConvAutoencoder(nn.Module):
    """Symmetric encoder/decoder for square inputs whose side is a multiple of 16.

    Input  : (B, 3, S, S)   normalised with ImageNet statistics
    Latent : (B, latent_channels, S/16, S/16)
    Output : (B, 3, S, S)   same normalised space, so loss is computed there
    """

    def __init__(self, base_channels: int = 32, latent_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            _conv_block(3, c),          # S   -> S/2
            _conv_block(c, c * 2),      # S/2 -> S/4
            _conv_block(c * 2, c * 4),  # S/4 -> S/8
            _conv_block(c * 4, c * 8),  # S/8 -> S/16
            nn.Conv2d(c * 8, latent_channels, kernel_size=1),  # channel squeeze
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, c * 8, kernel_size=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
            _upconv_block(c * 8, c * 4),
            _upconv_block(c * 4, c * 2),
            _upconv_block(c * 2, c),
            _upconv_block(c, 3, final=True),
        )
        self.latent_channels = latent_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel squared reconstruction error, summed over channels.

        Returns (B, H, W). We sum over channels rather than averaging so the map
        is a true squared-distance in colour space; averaging only rescales it
        and would make the number harder to relate to the training loss.
        """
        recon = self.forward(x)
        return ((x - recon) ** 2).sum(dim=1)


def build_autoencoder(cfg) -> ConvAutoencoder:
    return ConvAutoencoder(
        base_channels=cfg.base_channels,
        latent_channels=cfg.latent_channels,
    )


def gaussian_blur_map(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a batch of (B, H, W) anomaly maps with a separable Gaussian.

    WHY BLUR AT ALL: a single hot pixel is sensor noise, not a defect. Real
    defects occupy a *region*. Blurring makes the score respond to spatially
    coherent evidence and suppresses isolated spikes, which measurably improves
    AUROC. Both PaDiM and PatchCore do this; it is not a hack.

    We implement it with two 1-D convolutions instead of one 2-D kernel: a
    (k x k) Gaussian is separable, so this is O(k) per pixel instead of O(k^2),
    and it keeps everything on-device without an OpenCV dependency.
    """
    if sigma <= 0:
        return maps
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=maps.dtype, device=maps.device)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    x = maps.unsqueeze(1)  # (B, 1, H, W)
    # 'reflect' padding avoids the dark border that zero-padding would create,
    # which would otherwise read as "the edges are always normal".
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x.squeeze(1)
