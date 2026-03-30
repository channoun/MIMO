"""
LPIPS (Learned Perceptual Image Patch Similarity) wrapper.
Lower is better.

Requires: pip install lpips
"""
import torch
import torch.nn as nn

try:
    import lpips as _lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


class LPIPSMetric(nn.Module):
    """
    LPIPS metric wrapper.

    Input images should be in [-1, 1] (as per lpips library convention).
    """

    def __init__(self, net: str = "alex"):
        super().__init__()
        if not _LPIPS_AVAILABLE:
            raise ImportError(
                "lpips package not installed. Run: pip install lpips"
            )
        self.loss_fn = _lpips_lib.LPIPS(net=net)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1, img2: (B, 3, H, W) tensors in [-1, 1].

        Returns:
            lpips_val: Scalar mean LPIPS distance.
        """
        return self.loss_fn(img1, img2).mean()

    def per_sample(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Returns (B,) tensor of per-sample LPIPS distances."""
        return self.loss_fn(img1, img2).squeeze()
