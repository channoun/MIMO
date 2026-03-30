"""
Modified likelihood score for sub-Gaussian alpha-stable noise (Eq. in Section V).

The standard AWGN likelihood score:
    ∇_H ln p(Y | H, D) = -(Y - H @ X) X^H / sigma_n^2

is replaced by a posterior-expectation of Gaussian scores:
    ∇_H ln p(Y | H, D) = E_{p(A | Y, H, D)} [ -(Y - H @ X) X^H / (A * sigma_n^2) ]

The expectation over A is computed by Monte Carlo with L_A samples.

Per CLAUDE.md: "the original PVD likelihood score computation is recovered
term-by-term inside the expectation — only a one-dimensional integration over A
is added."

Everything else in pvd/pvd.py remains unchanged. Only this function is swapped in.
"""
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch.utils.checkpoint import checkpoint

from .stable_noise import SubGaussianStableNoise


def stable_likelihood_score(
    H_j: torch.Tensor,
    D_j: torch.Tensor,
    Y: torch.Tensor,
    f_gamma: nn.Module,
    S_theta_H: nn.Module,
    S_theta_D: nn.Module,
    sigma_H_j: float,
    sigma_D_j: float,
    noise_model: SubGaussianStableNoise,
    L_A: int = 20,
    use_checkpoint: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute likelihood scores under sub-Gaussian alpha-stable noise.

    Computes:
        ∇_{H_j} ln q(Y | Ψ_j) = E_{p(A|Y,Ψ)} [ ∇_{H_j} ln p(Y | Ψ_j, A) ]
        ∇_{D_j} ln q(Y | Ψ_j) = E_{p(A|Y,Ψ)} [ ∇_{D_j} ln p(Y | Ψ_j, A) ]

    where the inner term ∇ ln p(Y | Ψ_j, A) is the standard Gaussian likelihood
    score with sigma_n^2 replaced by A * sigma_n^2.

    Args:
        H_j:         (B, NrK, NtK) complex noisy channel.
        D_j:         (B, 3, H, W) noisy image.
        Y:           (B, NrK, T) received signal.
        f_gamma:     DJSCC encoder (frozen).
        S_theta_H:   Channel score network (frozen).
        S_theta_D:   Image score network (frozen).
        sigma_H_j:   Channel diffusion noise std.
        sigma_D_j:   Image diffusion noise std.
        noise_model: SubGaussianStableNoise instance.
        L_A:         Number of MC samples for A posterior.
        use_checkpoint: Use gradient checkpointing.

    Returns:
        grad_H: (B, NrK, NtK) complex gradient.
        grad_D: (B, 3, H, W) gradient.
    """
    B = H_j.shape[0]
    device = H_j.device

    # Compute Tweedie estimates (for A posterior)
    sigma_H_vec = torch.full((B,), sigma_H_j, dtype=torch.float32, device=device)
    sigma_D_vec = torch.full((B,), sigma_D_j, dtype=torch.float32, device=device)

    with torch.no_grad():
        H_in = torch.stack([H_j.real, H_j.imag], dim=1)
        score_H = S_theta_H(H_in, sigma_H_vec)
        H_hat = torch.complex(
            H_j.real + sigma_H_j ** 2 * score_H[:, 0],
            H_j.imag + sigma_H_j ** 2 * score_H[:, 1],
        )
        score_D = S_theta_D(D_j, sigma_D_vec)
        D_hat = D_j + sigma_D_j ** 2 * score_D
        X_hat_raw = f_gamma(D_hat)
        X_hat = X_hat_raw[:, 0] if X_hat_raw.dim() == 4 else X_hat_raw
        HX = torch.bmm(H_hat, X_hat)
        residual = Y - HX
        res_sq = (residual.abs() ** 2).sum(dim=(1, 2))  # (B,)

        # Sample A from posterior
        A_samples = noise_model.sample_A_posterior(res_sq, L_A=L_A)  # (B, L_A)

    # Average Gaussian likelihood scores over A samples
    # For each a, effective variance = a * sigma_n^2
    grad_H_accum = torch.zeros_like(H_j)
    grad_D_accum = torch.zeros_like(D_j)

    for l in range(L_A):
        A_l = A_samples[:, l]  # (B,)
        eff_var = A_l * (noise_model.sigma_n ** 2)  # (B,)

        H_j_c = H_j.clone().detach().requires_grad_(True)
        D_j_c = D_j.clone().detach().requires_grad_(True)

        sigma_H_v = torch.full((B,), sigma_H_j, dtype=torch.float32, device=device)
        sigma_D_v = torch.full((B,), sigma_D_j, dtype=torch.float32, device=device)

        H_in_c = torch.stack([H_j_c.real, H_j_c.imag], dim=1)
        score_H_c = S_theta_H(H_in_c, sigma_H_v)
        H_hat_c = torch.complex(
            H_j_c.real + sigma_H_j ** 2 * score_H_c[:, 0],
            H_j_c.imag + sigma_H_j ** 2 * score_H_c[:, 1],
        )
        score_D_c = S_theta_D(D_j_c, sigma_D_v)
        D_hat_c = D_j_c + sigma_D_j ** 2 * score_D_c

        if use_checkpoint:
            X_hat_c = checkpoint(f_gamma, D_hat_c, use_reentrant=False)
        else:
            X_hat_c = f_gamma(D_hat_c)
        if X_hat_c.dim() == 4:
            X_hat_c = X_hat_c[:, 0]

        HX_c = torch.bmm(H_hat_c, X_hat_c)
        residual_c = Y - HX_c
        res_sq_c = (residual_c.real**2 + residual_c.imag**2).sum(dim=(1, 2))
        loss = (res_sq_c / eff_var.clamp(min=1e-8)).sum()

        grads = torch.autograd.grad(loss, [H_j_c, D_j_c], allow_unused=True)
        gH = grads[0] if grads[0] is not None else torch.zeros_like(H_j)
        gD = grads[1] if grads[1] is not None else torch.zeros_like(D_j)

        grad_H_accum = grad_H_accum + (-gH) / L_A
        grad_D_accum = grad_D_accum + (-gD) / L_A

    return grad_H_accum, grad_D_accum
