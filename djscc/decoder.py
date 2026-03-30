"""
DJSCC Decoder g_β (mirror of DJSCCEncoder).

Used ONLY during codec training to reconstruct source images from channel symbols.
At inference time, the PVD blind receiver replaces this decoder entirely.

Architecture: inverse Swin Transformer with progressive upsampling,
mapping complex channel symbols back to image space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import WindowAttention, window_partition, window_reverse, SwinTransformerBlock


class PatchExpand(nn.Module):
    """Upsample by 2x via patch expanding (inverse of PatchMerging)."""

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 4 * dim // 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, H: int, W: int):
        """
        Args:
            x: [B, H*W, C]
        Returns:
            x: [B, 2H*2W, C//2]
        """
        B, L, C = x.shape
        x = self.expand(x)  # [B, H*W, 2C]
        x = x.view(B, H, W, -1)
        # Rearrange to [B, 2H, 2W, C//2]
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H * 2 * W, C // 2)
        x = self.norm(x)
        return x, 2 * H, 2 * W


class FinalPatchExpand(nn.Module):
    """Final upsampling back to full spatial resolution (patch_size x upscaling)."""

    def __init__(self, dim: int, patch_size: int = 4, out_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * patch_size * out_channels, bias=False)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C]
        Returns:
            out: [B, out_channels, H*patch_size, W*patch_size]
        """
        B, L, C = x.shape
        P = self.patch_size
        x = self.expand(x)  # [B, H*W, P^2 * out_channels]
        x = x.view(B, H, W, P, P, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C_out, H, P, W, P]
        x = x.view(B, self.out_channels, H * P, W * P)
        return x


class BasicDecoderLayer(nn.Module):
    """Swin Transformer decoder layer with optional upsampling."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int = 8, mlp_ratio: float = 4.,
                 dropout: float = 0., attn_dropout: float = 0.,
                 upsample: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout,
            )
            for i in range(depth)
        ])
        self.upsample = PatchExpand(dim) if upsample else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.upsample is not None:
            x, H, W = self.upsample(x, H, W)
        return x, H, W


class DJSCCDecoder(nn.Module):
    """
    DJSCC Decoder g_β: inverse of DJSCCEncoder.

    Reconstructs source image from channel symbols.
    Input:  X_flat ∈ C^{B×NtKT} (complex channel symbols)
    Output: D̂0 ∈ R^{B×3×256×256} (reconstructed image, values in [-1, 1])

    Only used during encoder training, NOT at inference time.
    """

    def __init__(
        self,
        out_channels: int = 3,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        patch_size: int = 4,
        num_symbols: int = 4608,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 6, 2, 2]   # reversed
        if num_heads is None:
            num_heads = [24, 12, 6, 3]  # reversed

        self.num_symbols = num_symbols
        self.patch_size = patch_size

        # Final spatial size after encoder's 3 downsamples: 8x8 with dim=768
        num_stages = len(depths)
        max_dim = embed_dim * (2 ** (num_stages - 1))   # 768
        init_spatial = (256 // patch_size) // (2 ** (num_stages - 1))  # 8

        self.init_dim = max_dim
        self.init_H = init_spatial
        self.init_W = init_spatial

        # Project channel symbols to initial feature map
        self.symbol_proj = nn.Sequential(
            nn.Linear(2 * num_symbols, max_dim * init_spatial * init_spatial),
            nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(max_dim)

        # Decoder stages (upsampling)
        # dims: 768->384->192->96
        self.layers = nn.ModuleList()
        for i, (d, nh) in enumerate(zip(depths, num_heads)):
            dim_i = embed_dim * (2 ** (num_stages - 1 - i))
            upsample = (i < num_stages - 1)
            layer = BasicDecoderLayer(
                dim=dim_i, depth=d, num_heads=nh, window_size=window_size,
                mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout,
                upsample=upsample,
            )
            self.layers.append(layer)

        # Final upsampling from patch grid back to pixel space
        self.final_expand = FinalPatchExpand(
            dim=embed_dim, patch_size=patch_size, out_channels=out_channels,
        )
        self.output_act = nn.Tanh()  # Map to [-1, 1]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Complex channel symbols [B, Nu, NtK, T] or [B, NtK, T].

        Returns:
            D_hat: Reconstructed image [B, 3, 256, 256].
        """
        # Handle multi-user: take first user's symbols
        if X.dim() == 4:
            X = X[:, 0]  # [B, NtK, T]

        B = X.shape[0]
        X_flat = X.reshape(B, -1)  # [B, NtK*T]

        # Concatenate real and imaginary parts
        x = torch.cat([X_flat.real, X_flat.imag], dim=-1)  # [B, 2*NtKT]

        # Project to feature space
        x = self.symbol_proj(x)  # [B, max_dim * H * W]
        x = x.view(B, self.init_H * self.init_W, self.init_dim)
        x = self.input_norm(x)

        H, W = self.init_H, self.init_W
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        # Final expand to image resolution
        D_hat = self.final_expand(x, H, W)   # [B, 3, 256, 256]
        D_hat = self.output_act(D_hat)
        return D_hat
