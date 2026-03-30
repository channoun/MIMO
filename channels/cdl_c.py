"""
CDL-C channel interface for Blind-MIMOSC.

CDL-C matrices are generated offline with QuaDRiGa (3GPP TR 38.901 CDL-C).
This module loads pre-saved matrices from .npy or .pt files.

If the file does not exist, a stub is returned and a warning is printed.
To generate CDL-C matrices:
  1. Install QuaDRiGa (MATLAB/Octave toolbox)
  2. Run 3GPP TR 38.901 CDL-C simulation
  3. Save complex channel matrices as numpy array of shape [N, Nr*K, Nt*K]
"""
import os
import math
import warnings
import numpy as np
import torch
from typing import Optional


def load_cdlc_channel(
    path: str,
    batch_size: int,
    Nr: int,
    Nt: int,
    K: int,
    device: torch.device = torch.device("cpu"),
    random_seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Load pre-generated CDL-C channel matrices.

    Args:
        path:       Path to .npy or .pt file with shape [N_total, Nr*K, Nt*K].
        batch_size: Number of samples to return.
        Nr, Nt, K:  Channel dimensions.
        device:     Computation device.
        random_seed: Optional seed for reproducible sampling.

    Returns:
        H0: (batch_size, 1, Nr*K, Nt*K) complex64 tensor.

    Raises:
        FileNotFoundError: If the CDL-C data file is not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CDL-C channel data not found at '{path}'.\n"
            "Generate it with QuaDRiGa (3GPP TR 38.901 CDL-C profile) and save "
            "as a numpy array of shape [N, Nr*K, Nt*K] (complex128 or complex64).\n"
            "See channels/cdl_c.py for details."
        )

    # Load data
    if path.endswith(".npy"):
        data = np.load(path)
        H_all = torch.from_numpy(data.astype(np.complex64)).to(device)
    else:
        H_all = torch.load(path, map_location=device, weights_only=False).to(torch.complex64)

    N_total = H_all.shape[0]
    expected_rows = Nr * K
    expected_cols = Nt * K
    if H_all.shape[1] != expected_rows or H_all.shape[2] != expected_cols:
        raise ValueError(
            f"CDL-C data shape mismatch. Expected [{N_total}, {expected_rows}, {expected_cols}], "
            f"got {list(H_all.shape)}."
        )

    # Random subset
    rng = torch.Generator(device=device)
    if random_seed is not None:
        rng.manual_seed(random_seed)
    idx = torch.randperm(N_total, generator=rng, device=device)[:batch_size]
    H0 = H_all[idx].unsqueeze(1)  # (batch, 1, Nr*K, Nt*K)
    return H0


def generate_synthetic_cdlc(
    batch_size: int,
    Nr: int,
    Nt: int,
    K: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate a synthetic CDL-C-like channel for testing when QuaDRiGa is unavailable.

    Uses a 3-cluster approximation with exponential power delay profile.
    This is NOT a faithful CDL-C implementation; use only for debugging.

    Returns:
        H0: (batch_size, 1, Nr*K, Nt*K) complex64 tensor.
    """
    warnings.warn(
        "Using synthetic CDL-C approximation (NOT faithful to 3GPP TR 38.901). "
        "For production experiments, use QuaDRiGa-generated matrices.",
        UserWarning,
        stacklevel=2,
    )
    num_clusters = 3
    # Power weights (exponential PDP)
    powers = torch.tensor([0.5, 0.3, 0.2], device=device)

    H0 = torch.zeros(batch_size, 1, Nr * K, Nt * K, dtype=torch.complex64, device=device)
    for k in range(K):
        H_k = torch.zeros(batch_size, Nr, Nt, dtype=torch.complex64, device=device)
        for c in range(num_clusters):
            # Random array steering vectors
            phase_r = torch.rand(batch_size, Nr, 1, device=device) * 2 * math.pi
            phase_t = torch.rand(batch_size, 1, Nt, device=device) * 2 * math.pi
            a_r = torch.complex(torch.cos(phase_r), torch.sin(phase_r))  # (B, Nr, 1)
            a_t = torch.complex(torch.cos(phase_t), torch.sin(phase_t))  # (B, 1, Nt)
            # Rank-1 component
            gain = torch.complex(
                torch.randn(batch_size, 1, 1, device=device),
                torch.randn(batch_size, 1, 1, device=device),
            ) / math.sqrt(2)
            H_k = H_k + powers[c].sqrt() * gain * (a_r @ a_t.transpose(-2, -1))  # (B, Nr, Nt)

        r0, r1 = k * Nr, (k + 1) * Nr
        c0, c1 = k * Nt, (k + 1) * Nt
        H0[:, 0, r0:r1, c0:c1] = H_k

    # Normalize to unit average Frobenius norm per block
    norm = H0.abs().pow(2).sum(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
    H0 = H0 / norm * math.sqrt(Nr * Nt * K)
    return H0
