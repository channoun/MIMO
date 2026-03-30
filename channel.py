"""
Channel models for Blind-MIMOSC.

Implements:
  - Rayleigh fading block-fading MIMO channel (Eq. 1-2 in paper)
  - CDL channel (requires QuaDRiGa data; uses saved matrices)
  - Channel application: Y = H0 * X + N
  - SNR utilities
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def generate_rayleigh_channel(
    batch_size: int,
    Nu: int,
    Nr: int,
    Nt: int,
    K: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate block-diagonal Rayleigh fading channel matrices.

    Each of K blocks is an independent Nr x Nt complex Gaussian realization.
    The compound channel H0 is block-diagonal: diag[H1, H2, ..., HK].

    Args:
        batch_size: Number of channel realizations.
        Nu: Number of users.
        Nr: Number of receive antennas.
        Nt: Number of transmit antennas per user.
        K: Number of transmission blocks.
        device: Computation device.

    Returns:
        H0: Compound block-diagonal channel [batch, Nu, Nr*K, Nt*K] (complex).
    """
    # Generate K independent fading blocks per user: [batch, Nu, K, Nr, Nt]
    real = torch.randn(batch_size, Nu, K, Nr, Nt, device=device)
    imag = torch.randn(batch_size, Nu, K, Nr, Nt, device=device)
    H_blocks = (real + 1j * imag) / np.sqrt(2)  # CN(0, 1)

    # Assemble block-diagonal matrix H0 [batch, Nu, Nr*K, Nt*K]
    H0 = torch.zeros(batch_size, Nu, Nr * K, Nt * K, dtype=torch.cfloat, device=device)
    for k in range(K):
        H0[:, :, k * Nr:(k + 1) * Nr, k * Nt:(k + 1) * Nt] = H_blocks[:, :, k]

    return H0


def load_cdlc_channel(
    path: str,
    batch_size: int,
    Nr: int,
    Nt: int,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load pre-generated CDL-C channel matrices from file.

    CDL channel matrices are generated offline with QuaDRiGa.
    The file should contain an array of shape [N_total, Nr*K, Nt*K] (complex).

    Args:
        path: Path to .npy or .pt file with channel matrices.
        batch_size: Number of samples to return.
        Nr, Nt, K: Channel dimensions.
        device: Computation device.

    Returns:
        H0: [batch, 1, Nr*K, Nt*K] complex channel matrices.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CDL channel data not found at {path}. "
            "Generate it with QuaDRiGa (3GPP TR 38.901 CDL-C model) "
            "and save as a numpy array of shape [N, Nr*K, Nt*K]."
        )
    if path.endswith(".npy"):
        data = np.load(path)
        H_all = torch.from_numpy(data).to(device)
    else:
        H_all = torch.load(path, map_location=device)

    # Random subset
    idx = torch.randperm(H_all.shape[0], device=device)[:batch_size]
    H0 = H_all[idx].unsqueeze(1)  # [batch, 1, Nr*K, Nt*K]
    return H0.to(torch.cfloat)


def apply_channel(
    H0: torch.Tensor,
    X: torch.Tensor,
    snr_db: float,
    power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply block-fading MIMO channel: Y = H0 * X + N (Eq. 2).

    Args:
        H0: Channel matrix [batch, Nu, Nr*K, Nt*K] (complex).
        X: Transmitted signal [batch, Nu, Nt*K, T] (complex).
        snr_db: Signal-to-noise ratio in dB.
        power: Expected signal power per element.

    Returns:
        Y: Received signal [batch, Nr*K, T] (complex).
        sigma_n: Noise standard deviation (scalar).
    """
    batch_size, Nu, NrK, NtK = H0.shape
    T = X.shape[-1]

    # Noise power: sigma_n^2 such that SNR = signal_power / sigma_n^2
    snr_linear = 10 ** (snr_db / 10.0)
    sigma_n = np.sqrt(power / snr_linear)

    # Sum received signals from all users: Y = sum_i H0_i * X_i + N
    # H0[:, i]: [batch, NrK, NtK], X[:, i]: [batch, NtK, T]
    HX = torch.zeros(batch_size, NrK, T, dtype=torch.cfloat, device=H0.device)
    for i in range(Nu):
        HX = HX + torch.bmm(H0[:, i], X[:, i])  # [batch, NrK, T]

    # Add AWGN: CN(0, sigma_n^2)
    noise_real = torch.randn_like(HX.real) * (sigma_n / np.sqrt(2))
    noise_imag = torch.randn_like(HX.imag) * (sigma_n / np.sqrt(2))
    N = torch.complex(noise_real, noise_imag)
    Y = HX + N

    return Y, sigma_n


def compute_snr(
    H0: torch.Tensor,
    X: torch.Tensor,
    N: torch.Tensor,
) -> float:
    """
    Compute actual channel SNR (Eq. 52):
    SNR = ||sum_i H0_i X_i||^2_F / ||N||^2_F
    """
    Nu = H0.shape[1]
    HX = sum(torch.bmm(H0[:, i], X[:, i]) for i in range(Nu))
    snr = (HX.abs() ** 2).sum() / (N.abs() ** 2).sum()
    return 10 * torch.log10(snr).item()


def normalize_channel(H0: torch.Tensor) -> torch.Tensor:
    """
    Normalize channel so that E[||H||^2_F] = Nr * Nt * K (unit average power per path).
    """
    Nr_K = H0.shape[-2]
    Nt_K = H0.shape[-1]
    scale = np.sqrt(Nr_K * Nt_K)
    return H0 / (H0.abs().mean(dim=(-2, -1), keepdim=True) * scale + 1e-8)


def noise_schedule_exponential(
    sigma_1: float,
    sigma_J: float,
    J: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Exponential interpolation noise schedule (Section III-C of paper):
    sigma_j = sigma_1 * (sigma_J / sigma_1)^(j/J)

    Returns sigma values for j = 0, 1, ..., J with sigma_0 = 0.
    """
    j_vals = torch.arange(1, J + 1, dtype=torch.float32, device=device)
    sigmas = sigma_1 * (sigma_J / sigma_1) ** (j_vals / J)
    # Prepend sigma_0 = 0
    sigmas = torch.cat([torch.zeros(1, device=device), sigmas])
    return sigmas  # shape [J+1]
