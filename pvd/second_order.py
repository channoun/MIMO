"""
Second-order trace score correction for PVD (Eq. 40, 42).

The calibrated likelihood variance (Eq. 42) requires the trace of the
Hessian of the log-prior:
    sigma_delta_N^2 = f(tr(∇^2_H ln q(H|σ_H)), tr(∇^2_D ln q(D|σ_D)))

This module computes these trace scores using the dedicated second-order
score networks.
"""
import torch
import torch.nn as nn
from typing import Tuple


def compute_trace_score_channel(
    H_j: torch.Tensor,
    sigma_j: float,
    s_theta_H: nn.Module,
) -> torch.Tensor:
    """
    Compute tr(∇^2_H ln q(H_j | sigma_j)) using the second-order score network.

    Args:
        H_j:       (B, NrK, NtK) complex channel latent.
        sigma_j:   Noise std at step j.
        s_theta_H: Second-order channel score network. Expects (B, 2, NrK, NtK).

    Returns:
        trace_H: (B,) scalar trace scores.
    """
    B = H_j.shape[0]
    H_in = torch.stack([H_j.real, H_j.imag], dim=1)
    sigma_vec = torch.full((B,), sigma_j, dtype=torch.float32, device=H_j.device)
    return s_theta_H(H_in, sigma_vec)


def compute_trace_score_image(
    D_j: torch.Tensor,
    sigma_j: float,
    s_theta_D: nn.Module,
) -> torch.Tensor:
    """
    Compute tr(∇^2_D ln q(D_j | sigma_j)) using the second-order score network.

    Args:
        D_j:       (B, 3, H, W) real image latent.
        sigma_j:   Noise std at step j.
        s_theta_D: Second-order image score network.

    Returns:
        trace_D: (B,) scalar trace scores.
    """
    B = D_j.shape[0]
    sigma_vec = torch.full((B,), sigma_j, dtype=torch.float32, device=D_j.device)
    return s_theta_D(D_j, sigma_vec)


def compute_sigma_delta_N(
    trace_H: torch.Tensor,
    trace_D: torch.Tensor,
    sigma_H_j: float,
    sigma_D_j: float,
    sigma_n: float,
    NrK: int,
    NtK: int,
    C: int,
    H_size: int,
    W_size: int,
) -> torch.Tensor:
    """
    Compute calibrated noise variance sigma_delta_N^2 (Eq. 42).

    sigma_delta_N^2 accounts for the Tweedie estimation error and is used
    to calibrate the likelihood score computation.

    Approximation (from Eq. 42 in the paper):
        sigma_delta_N^2 ≈ sigma_H_j^4 * |tr(∇^2_H ln q)| / (NrK * NtK)
                        + sigma_D_j^4 * |tr(∇^2_D ln q)| / (C * H * W)

    Args:
        trace_H:   (B,) trace scores for H.
        trace_D:   (B,) trace scores for D.
        sigma_H_j: Channel noise std.
        sigma_D_j: Image noise std.
        sigma_n:   Channel observation noise std.
        NrK, NtK:  Channel matrix dimensions.
        C, H_size, W_size: Image dimensions.

    Returns:
        sigma_delta_sq: (B,) calibrated noise variance.
    """
    # Per-element trace contribution
    h_contrib = sigma_H_j ** 4 * trace_H.abs() / (NrK * NtK)
    d_contrib = sigma_D_j ** 4 * trace_D.abs() / (C * H_size * W_size)
    sigma_delta_sq = (h_contrib + d_contrib).clamp(min=0)
    return sigma_delta_sq


def second_order_trace_correction(
    H_j: torch.Tensor,
    D_j: torch.Tensor,
    sigma_H_j: float,
    sigma_D_j: float,
    sigma_n: float,
    s_theta_H: nn.Module,
    s_theta_D: nn.Module,
) -> torch.Tensor:
    """
    Convenience wrapper: compute the full sigma_delta_N^2 correction.

    Args:
        H_j:       (B, NrK, NtK) complex channel latent.
        D_j:       (B, 3, H, W) real image latent.
        sigma_H_j: Channel noise std at step j.
        sigma_D_j: Image noise std at step j.
        sigma_n:   Base channel noise std.
        s_theta_H: Second-order channel score network.
        s_theta_D: Second-order image score network.

    Returns:
        sigma_delta_sq: (B,) calibrated noise variance.
    """
    NrK, NtK = H_j.shape[-2], H_j.shape[-1]
    C, H_size, W_size = D_j.shape[-3], D_j.shape[-2], D_j.shape[-1]

    trace_H = compute_trace_score_channel(H_j, sigma_H_j, s_theta_H)
    trace_D = compute_trace_score_image(D_j, sigma_D_j, s_theta_D)

    return compute_sigma_delta_N(
        trace_H, trace_D, sigma_H_j, sigma_D_j, sigma_n,
        NrK, NtK, C, H_size, W_size,
    )
