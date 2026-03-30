"""
DISTS (Deep Image Structure and Texture Similarity) metric.
Lower is better.

Implements a lightweight version using VGG features.
For the full implementation, use: pip install DISTS-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from DISTS_pytorch import DISTS as _DISTS
    _DISTS_AVAILABLE = True
except ImportError:
    _DISTS_AVAILABLE = False


class DISTSMetric(nn.Module):
    """
    DISTS metric wrapper.

    If DISTS_pytorch is available, uses the reference implementation.
    Otherwise falls back to a lightweight VGG-based approximation.

    Input images should be in [0, 1].
    """

    def __init__(self):
        super().__init__()
        if _DISTS_AVAILABLE:
            self.dists = _DISTS()
            self._use_ref = True
        else:
            # Lightweight fallback: use VGG16 feature MSE as proxy
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(vgg.features.children())[:16])
            for p in self.features.parameters():
                p.requires_grad_(False)
            self._use_ref = False
            self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1, img2: (B, 3, H, W) in [0, 1].

        Returns:
            dists_val: Scalar DISTS value.
        """
        if self._use_ref:
            return self.dists(img1, img2).mean()
        else:
            mean = self._mean.to(img1.device)
            std = self._std.to(img1.device)
            x1 = (img1 - mean) / std
            x2 = (img2 - mean) / std
            f1 = self.features(x1)
            f2 = self.features(x2)
            return F.mse_loss(f1, f2)
