"""
DJSCC Encoder based on SwinJSCC (Yang et al., 2023).

Maps source image D0 ∈ R^{H×W×C} to complex transmitted signal X ∈ C^{NtK×T}.
The encoder output satisfies the power constraint: (1/NtKT) ||X||^2_F ≤ P.

Architecture: Swin Transformer backbone with progressive downsampling,
followed by a linear projection to produce complex channel symbols.

Reference: [33] Yang et al., "SwinJSCC: Taming Swin Transformer for
Deep Joint Source-Channel Coding", 2023.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Window-based Multi-head Self-Attention (W-MSA / SW-MSA)
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.
    Args:
        x: [B, H, W, C]
        window_size: Window size M.
    Returns:
        windows: [num_windows*B, M, M, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.
    Args:
        windows: [num_windows*B, M, M, C]
    Returns:
        x: [B, H, W, C]
    """
    B_times_nW = windows.shape[0]
    B = int(B_times_nW / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention with relative position bias.
    Supports both W-MSA (shift=0) and SW-MSA (shift=window_size//2).
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table: (2M-1) x (2M-1) entries per head
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index for tokens inside a window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, M, M]
        coords_flat = coords.flatten(1)  # [2, M^2]
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, M^2, M^2]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [M^2, M^2]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [num_windows*B, M^2, C]
            mask: [num_windows, M^2, M^2] or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B_, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        rp_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        rp_bias = rp_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # [1, nH, M^2, M^2]
        attn = attn + rp_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with W-MSA or SW-MSA + MLP.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 4., dropout: float = 0.,
                 attn_dropout: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim, window_size=window_size, num_heads=num_heads,
            attn_drop=attn_dropout, proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C]
            H, W: Spatial dimensions.
        """
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition windows and compute attention
        x_windows = window_partition(x, self.window_size)  # [nW*B, M, M, C]
        x_windows = x_windows.view(-1, self.window_size ** 2, C)

        # Attention mask for SW-MSA
        attn_mask = self._compute_attn_mask(H, W, x.device) if self.shift_size > 0 else None
        attn_out = self.attn(x_windows, mask=attn_mask)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)

        # Reverse windows
        x = window_reverse(attn_out, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def _compute_attn_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask


class PatchEmbed(nn.Module):
    """Image to patch embedding: [B,C,H,W] -> [B, H'*W', embed_dim]."""

    def __init__(self, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        x = self.norm(x)
        return x, H_out, W_out


class PatchMerging(nn.Module):
    """Downsample by 2x via patch merging."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class BasicLayer(nn.Module):
    """A sequence of Swin Transformer blocks (W-MSA + SW-MSA pairs)."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int = 8, mlp_ratio: float = 4.,
                 dropout: float = 0., attn_dropout: float = 0.,
                 downsample: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


# ---------------------------------------------------------------------------
# DJSCC Encoder (SwinJSCC-based)
# ---------------------------------------------------------------------------

class DJSCCEncoder(nn.Module):
    """
    DJSCC Encoder f_γ based on SwinJSCC architecture.

    Maps source image D0 ∈ R^{B×3×256×256} to complex channel symbols
    X ∈ C^{B×Nu×NtK×T} satisfying the power constraint.

    Training: paired with DJSCCDecoder to minimize ||D0 - g_β(f_γ(D0))||^2_F.
    Inference: X is fed into the MIMO channel.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        patch_size: int = 4,
        num_symbols: int = 4608,    # NtK*T total complex symbols
        Nt: int = 1,
        K: int = 192,
        T: int = 24,
        Nu: int = 1,                # Users (for encoder, Nu=1 typically)
        power: float = 1.0,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.num_symbols = num_symbols
        self.Nt = Nt
        self.K = K
        self.T = T
        self.Nu = Nu
        self.power = power

        # Patch embedding: 256x256 -> 64x64, embed_dim=96
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)

        # Swin Transformer stages with progressive downsampling
        # After patch embedding: 64x64 with embed_dim=96
        # Stage dims: 96, 192, 384, 768
        self.layers = nn.ModuleList()
        for i, (d, nh) in enumerate(zip(depths, num_heads)):
            dim_i = embed_dim * (2 ** i)
            layer = BasicLayer(
                dim=dim_i, depth=d, num_heads=nh, window_size=window_size,
                mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout,
                downsample=(i < len(depths) - 1),
            )
            self.layers.append(layer)

        # Final feature dimension after all layers
        # With 3 downsamples: 64x64 -> 32x32 -> 16x16 -> 8x8
        # Feature dim: 96 * 2^(len(depths)-1) = 96*8 = 768
        final_dim = embed_dim * (2 ** (len(depths) - 1))
        final_spatial = (256 // patch_size) // (2 ** (len(depths) - 1))  # 8
        self.final_feat_size = final_dim * final_spatial * final_spatial  # 768*8*8 = 49152

        # Project flattened features to complex channel symbols
        # Output: num_symbols real + num_symbols imaginary = 2*num_symbols
        self.channel_proj = nn.Sequential(
            nn.LayerNorm(self.final_feat_size),
            nn.Linear(self.final_feat_size, 2 * num_symbols),
        )

    def forward(self, D0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            D0: Source image [B, 3, 256, 256], values in [-1, 1].

        Returns:
            X: Transmitted signal [B, Nu, NtK, T] complex, power-normalized.
        """
        B = D0.shape[0]

        # Swin Transformer feature extraction
        x, H, W = self.patch_embed(D0)
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        # Flatten: [B, H*W*C]
        x = x.view(B, -1)

        # Project to channel symbols (real + imag concatenated)
        x = self.channel_proj(x)  # [B, 2*num_symbols]

        # Split into real and imaginary parts
        real = x[:, :self.num_symbols]
        imag = x[:, self.num_symbols:]
        X_flat = torch.complex(real, imag)  # [B, NtK*T]

        # Reshape to [B, Nu, NtK, T]
        X = X_flat.view(B, self.Nt * self.K, self.T)
        X = X.unsqueeze(1).expand(-1, self.Nu, -1, -1)

        # Power normalization: (1/NtKT) ||X||^2_F = power
        X = self._normalize_power(X)

        return X

    def _normalize_power(self, X: torch.Tensor) -> torch.Tensor:
        """
        Normalize so that average power per element = self.power.
        (1/NtKT) ||X_i||^2_F = P for each user i and batch sample.
        """
        B, Nu, NtK, T = X.shape
        # Compute per-sample power
        power_actual = (X.abs() ** 2).mean(dim=(-2, -1), keepdim=True)  # [B, Nu, 1, 1]
        power_actual = power_actual.clamp(min=1e-8)
        X_norm = X * (self.power / power_actual).sqrt()
        return X_norm
