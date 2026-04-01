"""
Likelihood score computation for PVD (Eq. 43).

Calibrated likelihood score:
    ∇_{H_j} ln q(Y | Ψ_j) = ∇_{H_j} [
        -||Y - H_hat_0|j @ f_γ(D_hat_0|j)||^2_F / (sigma_delta_N^2 + sigma_n^2)
    ]

The gradient is computed via implicit differentiation through the Tweedie
estimates, using PyTorch autograd.

Jacobian-vector products through the encoder f_γ are computed implicitly
(never forming the full Jacobian, which is O(600M) entries).
"""
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple
from torch.utils.checkpoint import checkpoint


def _residual_norm_sq(
    H_hat: torch.Tensor,
    X_hat: torch.Tensor,
    Y: torch.Tensor,
    effective_var: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ||Y - H_hat @ X_hat||^2_F / (sigma_eff^2) summed over the batch.

    Args:
        H_hat:        (B, NrK, NtK) complex Tweedie channel estimate.
        X_hat:        (B, NtK, T) complex transmitted signal estimate.
        Y:            (B, NrK, T) complex received signal.
        effective_var: (B,) effective noise variance = sigma_delta^2 + sigma_n^2.

    Returns:
        Scalar loss (used only for gradient computation).
    """
    HX = torch.bmm(H_hat, X_hat)  # (B, NrK, T)
    residual = Y - HX              # (B, NrK, T)
    res_sq = (residual.real ** 2 + residual.imag ** 2)  # (B, NrK, T)
    # Sum over spatial dims, divide by effective variance per batch element
    loss = (res_sq.sum(dim=(1, 2)) / effective_var.clamp(min=1e-8)).sum()
    return loss


def likelihood_score(
    H_j: torch.Tensor,
    D_j: torch.Tensor,
    Y: torch.Tensor,
    f_gamma: nn.Module,
    S_theta_H: nn.Module,
    S_theta_D: nn.Module,
    sigma_H_j: float,
    sigma_D_j: float,
    effective_var: torch.Tensor,
    use_checkpoint: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute likelihood scores ∇_{H_j} ln q(Y|Ψ_j) and ∇_{D_j} ln q(Y|Ψ_j).

    Uses autograd through the Tweedie estimates. The encoder Jacobian is never
    materialized; instead we rely on backprop through f_γ.

    Args:
        H_j:          (B, NrK, NtK) complex noisy channel at step j.
        D_j:          (B, 3, H, W) real noisy image at step j.
        Y:            (B, NrK, T) complex received signal.
        f_gamma:      DJSCC encoder (frozen during PVD).
        S_theta_H:    Channel score network (frozen).
        S_theta_D:    Image score network (frozen).
        sigma_H_j:    Channel noise std at step j.
        sigma_D_j:    Image noise std at step j.
        effective_var: (B,) effective noise variance.
        use_checkpoint: Use gradient checkpointing for f_gamma.

    Returns:
        grad_H: (B, NrK, NtK) complex gradient w.r.t. H_j.
        grad_D: (B, 3, H, W) gradient w.r.t. D_j.
    """
    B = H_j.shape[0]

    # We need gradients w.r.t. H_j and D_j
    H_j_c = H_j.clone().detach().requires_grad_(True)
    D_j_c = D_j.clone().detach().requires_grad_(True)

    # Tweedie estimates (differentiable through the score networks)
    sigma_H_vec = torch.full((B,), sigma_H_j, dtype=torch.float32, device=H_j.device)
    sigma_D_vec = torch.full((B,), sigma_D_j, dtype=torch.float32, device=D_j.device)

    H_in = torch.stack([H_j_c.real, H_j_c.imag], dim=1)
    score_H = S_theta_H(H_in, sigma_H_vec)  # (B, 2, NrK, NtK)
    H_hat_real = H_j_c.real + sigma_H_j ** 2 * score_H[:, 0]
    H_hat_imag = H_j_c.imag + sigma_H_j ** 2 * score_H[:, 1]
    H_hat = torch.complex(H_hat_real, H_hat_imag)  # (B, NrK, NtK)

    score_D = S_theta_D(D_j_c, sigma_D_vec)  # (B, 3, H, W)
    D_hat = D_j_c + sigma_D_j ** 2 * score_D  # (B, 3, H, W)

    # print("score D: ", score_D)
    # print("score H: ", score_H)

    # Encode D_hat → X_hat
    if use_checkpoint:
        X_hat_raw = checkpoint(f_gamma, D_hat, use_reentrant=False)
    else:
        X_hat_raw = f_gamma(D_hat)
    # Take first user (or average over users)
    if X_hat_raw.dim() == 4:
        X_hat = X_hat_raw[:, 0]  # (B, NtK, T)
    else:
        X_hat = X_hat_raw  # (B, NtK, T)

    # Residual loss
    print("effective_var: ", effective_var)
    print("H_hat norm:", H_hat.abs().mean().item(), H_hat.abs().max().item())
    print("X_hat norm:", X_hat.abs().mean().item(), X_hat.abs().max().item())
    print("Y norm:", Y.abs().mean().item(), Y.abs().max().item())
    loss = _residual_norm_sq(H_hat, X_hat, Y, effective_var)
    print("loss: ", loss)

    # Compute gradients
    grads = torch.autograd.grad(loss, [H_j_c, D_j_c], allow_unused=True)
    grad_H_real = grads[0] if grads[0] is not None else torch.zeros_like(H_j)
    grad_D = grads[1] if grads[1] is not None else torch.zeros_like(D_j)

    # Convert to complex gradient for H (Wirtinger: conj of autograd grad)
    # For real-valued loss and complex input, autograd gives the Wirtinger derivative.
    # We want the direction of steepest descent in C^n: use the conjugate.
    if H_j.is_complex():
        # grad_H_real is actually treating H_j_c as real-valued tensor
        # (complex tensors stored as real). The score is real+imag stacked.
        NrK, NtK = H_j.shape[-2], H_j.shape[-1]
        # grad_H_real shape: (B, NrK, NtK) complex
        grad_H = grad_H_real
    else:
        grad_H = grad_H_real

    # Negate: we want the gradient of -loss (the score points uphill)
    return -grad_H, -grad_D


def likelihood_score_simple(
    H_hat: torch.Tensor,
    X_hat: torch.Tensor,
    Y: torch.Tensor,
    H_j: torch.Tensor,
    D_j: torch.Tensor,
    sigma_H_j: float,
    sigma_D_j: float,
    effective_var: torch.Tensor,
    f_gamma: nn.Module,
    S_theta_H: nn.Module,
    S_theta_D: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified likelihood score using detached Tweedie estimates.

    This is a computationally cheaper variant where H_hat and D_hat
    are treated as constants (no gradient through the score networks).
    Uses only the Jacobian of f_γ at D_hat.

    Suitable when the score network Jacobian correction is small.
    """
    B = H_j.shape[0]
    H_j_c = H_j.detach().clone().requires_grad_(True)
    D_j_c = D_j.detach().clone().requires_grad_(True)

    # Tweedie estimates computed with grad tracking
    sigma_H_vec = torch.full((B,), sigma_H_j, dtype=torch.float32, device=H_j.device)
    sigma_D_vec = torch.full((B,), sigma_D_j, dtype=torch.float32, device=D_j.device)

    H_in = torch.stack([H_j_c.real, H_j_c.imag], dim=1)
    with torch.no_grad():
        score_H = S_theta_H(H_in, sigma_H_vec)
        print("score h: ", score_H)
    H_hat_real = H_j_c.real + sigma_H_j ** 2 * score_H[:, 0].detach()
    H_hat_imag = H_j_c.imag + sigma_H_j ** 2 * score_H[:, 1].detach()
    H_hat_grad = torch.complex(H_hat_real, H_hat_imag)

    with torch.no_grad():
        score_D = S_theta_D(D_j_c, sigma_D_vec)
        print("score d: ", score_D)
    D_hat_grad = D_j_c + sigma_D_j ** 2 * score_D.detach()

    X_hat_grad = f_gamma(D_hat_grad)
    if X_hat_grad.dim() == 4:
        X_hat_grad = X_hat_grad[:, 0]

    print("effective_var: ", effective_var)
    loss = _residual_norm_sq(H_hat_grad, X_hat_grad, Y, effective_var)
    print("loss:",  loss)
    grads = torch.autograd.grad(loss, [H_j_c, D_j_c], allow_unused=True)
    gH = grads[0] if grads[0] is not None else torch.zeros_like(H_j)
    gD = grads[1] if grads[1] is not None else torch.zeros_like(D_j)
    return -gH, -gD
