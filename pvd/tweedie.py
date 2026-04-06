"""
Tweedie denoising estimates for PVD.

Eq. 38 from the paper:
    H_hat_0|j = H_j + sigma_H_j^2 * S_theta_H(H_j, sigma_H_j)
    D_hat_0|j = D_j + sigma_D_j^2 * S_theta_D(D_j, sigma_D_j)

These are MMSE estimates of H0 (or D0) given the noisy latent H_j (or D_j).
"""
import torch
import torch.nn as nn
from typing import Callable


def tweedie_estimate(
    x_j: torch.Tensor,
    sigma_j: float,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Generic Tweedie MMSE estimate (Eq. 38):
        x_hat_0|j = x_j + sigma_j^2 * score_fn(x_j, sigma_j)

    Args:
        x_j:      Noisy latent at step j (any shape with batch dim).
        sigma_j:  Scalar noise standard deviation at step j.
        score_fn: Callable (x, sigma_vec) → score.
                  sigma_vec is a (B,) tensor of the same value.

    Returns:
        x_hat_0: Tweedie estimate of x_0 (same shape as x_j).
    """
    B = x_j.shape[0]
    sigma_vec = torch.full((B,), sigma_j, dtype=torch.float32, device=x_j.device)
    score = score_fn(x_j, sigma_vec)
    return x_j + sigma_j ** 2 * score


def tweedie_channel(
    H_j: torch.Tensor,
    sigma_j: float,
    S_theta_H: nn.Module,
) -> torch.Tensor:
    """
    Tweedie estimate for the channel H.

    H_j is complex-valued (B, NrK, NtK). The score network expects real/imag
    stacked as (B, 2, NrK, NtK).

    Returns:
        H_hat_0: (B, NrK, NtK) complex tensor.
    """
    B = H_j.shape[0]
    # Channel network trained on H_j/sigma_j; predicts -epsilon (not the score).
    # Tweedie: H_hat = H_j + sigma * net(H_j/sigma)
    H_j_real = torch.stack([H_j.real, H_j.imag], dim=1) / sigma_j  # normalize input
    sigma_vec = torch.full((B,), sigma_j, dtype=torch.float32, device=H_j.device)
    eps_pred = S_theta_H(H_j_real, sigma_vec)  # (B, 2, NrK, NtK) ≈ -epsilon
    H_hat_real = H_j.real + sigma_j * eps_pred[:, 0]
    H_hat_imag = H_j.imag + sigma_j * eps_pred[:, 1]
    return torch.complex(H_hat_real, H_hat_imag)


def tweedie_image(
    D_j: torch.Tensor,
    sigma_j: float,
    S_theta_D: nn.Module,
) -> torch.Tensor:
    """
    Tweedie estimate for the image D.

    NCSNpp is an epsilon predictor: output ≈ -eps.
    Tweedie: D_hat = D_j + sigma² * score
           = D_j + sigma² * (eps_pred / sigma)
           = D_j + sigma * eps_pred

    Args:
        D_j:      (B, 3, H, W) real noisy image at step j.
        sigma_j:  Noise std at step j.
        S_theta_D: Score network, expects (D, sigma_vec). Output ≈ -eps.

    Returns:
        D_hat_0: (B, 3, H, W) Tweedie estimate.
    """
    B = D_j.shape[0]
    sigma_vec = torch.full((B,), sigma_j, dtype=torch.float32, device=D_j.device)
    eps_pred = S_theta_D(D_j, sigma_vec)  # ≈ -eps
    return D_j + sigma_j * eps_pred
