"""
Parallel Variational Diffusion (PVD) — Algorithm 1 from the paper.

Blind joint recovery of MIMO channel H0 and source data D0 from Y.

Model: Y = H0 @ f_γ(D0) + N,   N ~ CN(0, sigma_n^2 I)

PVD runs J reverse diffusion steps, each with J_in inner iterations
that update the variational means H_hat_j and D_hat_j using:
  1. Transition score (Langevin gradient from diffusion)
  2. Prior score (from trained score networks)
  3. Likelihood score (from Eq. 43, via autograd through f_γ)

Parameters (from paper Table in Section IV):
  J = 30, J_in = 20, L = 1
  sigma_H_J = sigma_D_J = 100
  sigma_H_1 = sigma_D_1 = 0.01
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from tqdm import tqdm

from .tweedie import tweedie_channel, tweedie_image
from .second_order import (
    compute_trace_score_channel,
    compute_trace_score_image,
    compute_sigma_delta_N,
)
from .likelihood import likelihood_score
from channels.rayleigh import noise_schedule_exponential


class PVDSolver:
    """
    PVD blind receiver.

    Jointly recovers H0 and D0 from the received signal Y.

    Usage:
        pvd = PVDSolver(
            f_gamma=encoder, S_theta_H=ch_score, S_theta_D=img_score,
            s_theta_H=ch_score_2nd, s_theta_D=img_score_2nd,
            sigma_n=sigma_n, Nr=4, Nt=1, K=192, T=24, device=device,
        )
        H_hat, D_hat = pvd.solve(Y)
    """

    def __init__(
        self,
        f_gamma: nn.Module,
        S_theta_H: nn.Module,
        S_theta_D: nn.Module,
        s_theta_H: Optional[nn.Module],
        s_theta_D: Optional[nn.Module],
        sigma_n: float,
        Nr: int,
        Nt: int,
        K: int,
        T: int,
        Nu: int = 1,
        J: int = 30,
        J_in: int = 20,
        L: int = 1,
        sigma_H_1: float = 0.01,
        sigma_H_J: float = 100.0,
        sigma_D_1: float = 0.01,
        sigma_D_J: float = 100.0,
        zeta_H: float = 1.0,
        zeta_D: float = 1.0,
        device: torch.device = torch.device("cpu"),
        use_second_order: bool = True,
        use_checkpoint: bool = True,
    ):
        self.f_gamma = f_gamma.eval()
        self.S_theta_H = S_theta_H.eval()
        self.S_theta_D = S_theta_D.eval()
        self.s_theta_H = s_theta_H.eval() if s_theta_H is not None else None
        self.s_theta_D = s_theta_D.eval() if s_theta_D is not None else None

        self.sigma_n = sigma_n
        self.Nr, self.Nt, self.K, self.T, self.Nu = Nr, Nt, K, T, Nu
        self.J, self.J_in, self.L = J, J_in, L
        self.zeta_H, self.zeta_D = zeta_H, zeta_D
        self.device = device
        self.use_second_order = use_second_order
        self.use_checkpoint = use_checkpoint

        # Noise schedules: shape (J+1,), sigmas[0]=0, sigmas[j]=sigma_j
        self.sigmas_H = noise_schedule_exponential(sigma_H_1, sigma_H_J, J, device)
        self.sigmas_D = noise_schedule_exponential(sigma_D_1, sigma_D_J, J, device)

        # Freeze all networks
        for net in [self.f_gamma, self.S_theta_H, self.S_theta_D]:
            for p in net.parameters():
                p.requires_grad_(False)
        if self.s_theta_H is not None:
            for p in self.s_theta_H.parameters():
                p.requires_grad_(False)
        if self.s_theta_D is not None:
            for p in self.s_theta_D.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def _init_latents(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize H_J and D_J by sampling from the prior at max noise level.
        """
        NrK = self.Nr * self.K
        NtK = self.Nt * self.K
        sigma_H_J = self.sigmas_H[-1].item()
        sigma_D_J = self.sigmas_D[-1].item()

        # H_J ~ CN(0, sigma_H_J^2 * I)
        H_J = torch.complex(
            torch.randn(batch_size, NrK, NtK, device=self.device) * sigma_H_J / math.sqrt(2),
            torch.randn(batch_size, NrK, NtK, device=self.device) * sigma_H_J / math.sqrt(2),
        )
        # D_J ~ N(0, sigma_D_J^2 * I)
        D_J = torch.randn(batch_size, 3, 256, 256, device=self.device) * sigma_D_J

        return H_J, D_J

    def _effective_var(
        self,
        H_j: torch.Tensor,
        D_j: torch.Tensor,
        sigma_H_j: float,
        sigma_D_j: float,
    ) -> torch.Tensor:
        """
        Compute effective variance for likelihood score (Eq. 42).
        Falls back to sigma_n^2 if second-order networks not available.
        """
        B = H_j.shape[0]
        base_var = torch.full((B,), self.sigma_n ** 2, device=self.device)

        if not self.use_second_order or self.s_theta_H is None or self.s_theta_D is None:
            return base_var

        with torch.no_grad():
            trace_H = compute_trace_score_channel(H_j, sigma_H_j, self.s_theta_H)
            trace_D = compute_trace_score_image(D_j, sigma_D_j, self.s_theta_D)

        sigma_delta_sq = compute_sigma_delta_N(
            trace_H, trace_D, sigma_H_j, sigma_D_j, self.sigma_n,
            NrK=self.Nr * self.K, NtK=self.Nt * self.K,
            C=3, H_size=256, W_size=256,
        )
        return (base_var + sigma_delta_sq).clamp(min=self.sigma_n ** 2)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stat(t: torch.Tensor, name: str) -> str:
        """Return a compact stats string for a real or complex tensor."""
        v = t.abs() if t.is_complex() else t.abs()
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        flag = " *** NaN ***" if has_nan else (" *** Inf ***" if has_inf else "")
        return (f"{name:22s}  mean={v.mean().item():12.4e}  "
                f"max={v.max().item():12.4e}{flag}")

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(
        self,
        Y: torch.Tensor,
        verbose: bool = True,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run PVD Algorithm 1.

        Args:
            Y:       (B, NrK, T) complex received signal.
            verbose: Show progress bar.
            debug:   Print per-step diagnostics to stdout.

        Returns:
            H_hat: (B, NrK, NtK) complex channel estimate.
            D_hat: (B, 3, 256, 256) reconstructed image.
        """
        B = Y.shape[0]
        H_j, D_j = self._init_latents(B)

        if debug:
            print("\n" + "="*70)
            print("PVD DEBUG — pre-loop state")
            print("="*70)
            print(self._stat(Y,   "Y (received)"))
            print(self._stat(H_j, "H_j (init)"))
            print(self._stat(D_j, "D_j (init)"))
            print(f"{'sigma_n':22s}  {self.sigma_n:.4e}")
            print(f"{'sigmas_H range':22s}  [{self.sigmas_H[1].item():.4e}, {self.sigmas_H[-1].item():.4e}]")
            print(f"{'sigmas_D range':22s}  [{self.sigmas_D[1].item():.4e}, {self.sigmas_D[-1].item():.4e}]")
            print(f"{'zeta_H / zeta_D':22s}  {self.zeta_H} / {self.zeta_D}")

        outer_iter = range(self.J, 0, -1)
        if verbose and not debug:
            outer_iter = tqdm(list(outer_iter), desc="PVD", unit="step")

        for j in outer_iter:
            sigma_H_j = self.sigmas_H[j].item()
            sigma_H_j1 = self.sigmas_H[j - 1].item()
            sigma_D_j = self.sigmas_D[j].item()
            sigma_D_j1 = self.sigmas_D[j - 1].item()

            # Step sizes (Eq. in Section III-C)
            eps_H = self.zeta_H * (sigma_H_j1 ** 2 - sigma_H_j ** 2)  # Note: j-1 < j
            eps_D = self.zeta_D * (sigma_D_j1 ** 2 - sigma_D_j ** 2)  # negative → descending

            # Use absolute step sizes (diffusion goes from high to low sigma)
            eps_H = abs(eps_H)
            eps_D = abs(eps_D)

            if debug:
                print(f"\n{'─'*70}")
                print(f"  OUTER j={j:3d}  σ_H={sigma_H_j:.4e}  σ_D={sigma_D_j:.4e}"
                      f"  ε_H={eps_H:.4e}  ε_D={eps_D:.4e}")
                print(f"  {'H_j (start of outer)':22s}  "
                      f"mean={H_j.abs().mean().item():.4e}  "
                      f"max={H_j.abs().max().item():.4e}")

            for inner_i in range(self.J_in):
                # Effective variance for likelihood
                eff_var = self._effective_var(H_j, D_j, sigma_H_j, sigma_D_j)

                # Likelihood scores (requires grad)
                with torch.enable_grad():
                    grad_H_lik, grad_D_lik = likelihood_score(
                        H_j, D_j, Y,
                        self.f_gamma, self.S_theta_H, self.S_theta_D,
                        sigma_H_j, sigma_D_j, eff_var,
                        use_checkpoint=self.use_checkpoint,
                    )

                    if torch.isnan(grad_H_lik).any() or torch.isnan(grad_D_lik).any():
                        print(f"[j={j} inner={inner_i}] NaN in likelihood gradients")
                        if debug:
                            print(self._stat(eff_var,    "  eff_var"))
                            print(self._stat(H_j,        "  H_j"))
                            print(self._stat(D_j,        "  D_j"))
                            print(self._stat(grad_H_lik, "  grad_H_lik (NaN)"))
                            print(self._stat(grad_D_lik, "  grad_D_lik"))
                        break

                # Prior scores (no grad needed)
                with torch.no_grad():
                    B_size = H_j.shape[0]
                    sigma_H_vec = torch.full((B_size,), sigma_H_j, device=self.device)
                    sigma_D_vec = torch.full((B_size,), sigma_D_j, device=self.device)

                    # Channel network trained on H_j/sigma_j; predicts -epsilon (not score).
                    # Actual score = net_output / sigma_H_j.
                    H_in = torch.stack([H_j.real, H_j.imag], dim=1) / sigma_H_j
                    score_H_prior = self.S_theta_H(H_in, sigma_H_vec)  # (B, 2, NrK, NtK) ≈ -eps
                    score_H_complex = torch.complex(
                        score_H_prior[:, 0] / sigma_H_j,
                        score_H_prior[:, 1] / sigma_H_j,
                    )  # actual score ∇ log p(H_j)

                    score_D_prior = self.S_theta_D(D_j, sigma_D_vec)   # (B, 3, H, W)

                # Stochastic Langevin update (Eq. 36) with L=1 sample:
                noise_H = torch.complex(
                    torch.randn_like(H_j.real) * math.sqrt(eps_H),
                    torch.randn_like(H_j.imag) * math.sqrt(eps_H),
                )
                noise_D = torch.randn_like(D_j) * math.sqrt(eps_D)

                # Full score for H: prior + likelihood
                total_score_H_real = score_H_complex.real + grad_H_lik.real
                total_score_H_imag = score_H_complex.imag + grad_H_lik.imag

                # Debug: print the first inner iteration of every outer step
                if debug and inner_i == 0:
                    prior_contrib = eps_H * score_H_complex.abs().mean().item()
                    lik_contrib   = eps_H * grad_H_lik.abs().mean().item()
                    noise_contrib = noise_H.abs().mean().item()
                    print(f"  {'eff_var (mean)':22s}  {eff_var.mean().item():.4e}")
                    print(f"  {'score_H raw (-eps)':22s}  "
                          f"mean={score_H_prior.abs().mean().item():.4e}  "
                          f"max={score_H_prior.abs().max().item():.4e}")
                    print(f"  {'score_H actual':22s}  "
                          f"mean={score_H_complex.abs().mean().item():.4e}  "
                          f"max={score_H_complex.abs().max().item():.4e}")
                    print(f"  {'grad_H_lik':22s}  "
                          f"mean={grad_H_lik.abs().mean().item():.4e}  "
                          f"max={grad_H_lik.abs().max().item():.4e}")
                    print(f"  {'score_D prior':22s}  "
                          f"mean={score_D_prior.abs().mean().item():.4e}  "
                          f"max={score_D_prior.abs().max().item():.4e}")
                    print(f"  {'grad_D_lik':22s}  "
                          f"mean={grad_D_lik.abs().mean().item():.4e}  "
                          f"max={grad_D_lik.abs().max().item():.4e}")
                    print(f"  update H contributions:")
                    print(f"    eps_H*prior  = {prior_contrib:.4e}")
                    print(f"    eps_H*lik    = {lik_contrib:.4e}")
                    print(f"    noise        = {noise_contrib:.4e}")

                # Update variational mean (Eq. 36)
                H_j = torch.complex(
                    H_j.real + eps_H * total_score_H_real + noise_H.real,
                    H_j.imag + eps_H * total_score_H_imag + noise_H.imag,
                )
                D_j = D_j + eps_D * (score_D_prior + grad_D_lik) + noise_D

                # Check for blowup / NaN
                h_max = H_j.abs().max().item()
                if torch.isnan(H_j).any() or torch.isnan(D_j).any() or h_max > 1e6:
                    print(f"[j={j} inner={inner_i}] "
                          f"{'NaN' if torch.isnan(H_j).any() else 'BLOWUP'} "
                          f"in H_j  max={h_max:.4e}")
                    if debug:
                        print(self._stat(score_H_complex, "  score_H_complex"))
                        print(self._stat(grad_H_lik,      "  grad_H_lik"))
                        print(self._stat(noise_H,         "  noise_H"))
                        print(f"  eps_H*prior={prior_contrib:.4e}  "
                              f"eps_H*lik={lik_contrib:.4e}  "
                              f"noise={noise_contrib:.4e}")
                    break

        # Final Tweedie estimates
        with torch.no_grad():
            sigma_H_vec = torch.full((B,), self.sigmas_H[1].item(), device=self.device)
            sigma_D_vec = torch.full((B,), self.sigmas_D[1].item(), device=self.device)

            sigma_H_1 = self.sigmas_H[1].item()
            H_in = torch.stack([H_j.real, H_j.imag], dim=1) / sigma_H_1
            score_H = self.S_theta_H(H_in, sigma_H_vec)
            # Tweedie: H_hat = H_j + sigma * net(H_j/sigma)
            H_hat = torch.complex(
                H_j.real + sigma_H_1 * score_H[:, 0],
                H_j.imag + sigma_H_1 * score_H[:, 1],
            )

            score_D = self.S_theta_D(D_j, sigma_D_vec)
            D_hat_norm = D_j + self.sigmas_D[1].item() ** 2 * score_D
            D_hat = D_hat_norm.clamp(-1, 1)

        return H_hat, D_hat
