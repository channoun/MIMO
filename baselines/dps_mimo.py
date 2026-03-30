"""
DPS-MIMO baseline: Diffusion Posterior Sampling with pilot-based LMMSE.

Uses the trained image score network as a prior, but with pilots for
channel estimation. This is a non-blind baseline.

Reference: Chung et al., "Diffusion Posterior Sampling for General
Noisy Inverse Problems." ICLR 2023.
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from tqdm import tqdm

from channels.rayleigh import lmmse_channel_estimate, noise_schedule_exponential


class DPSMIMOBaseline:
    """
    DPS-MIMO: non-blind image reconstruction using diffusion posterior sampling.

    Channel is estimated from pilots (LMMSE), then image D0 is recovered
    by running DPS with the image score network.

    Args:
        encoder:    Trained DJSCCEncoder.
        S_theta_D:  Trained image score network.
        Nr, Nt, K, T: Channel dimensions.
        Nu:         Number of users.
        J:          Number of diffusion steps.
        zeta_D:     DPS step size multiplier.
        n_pilots:   Number of pilot symbols.
    """

    def __init__(
        self,
        encoder: nn.Module,
        S_theta_D: nn.Module,
        Nr: int,
        Nt: int,
        K: int,
        T: int,
        Nu: int = 1,
        J: int = 30,
        sigma_D_1: float = 0.01,
        sigma_D_J: float = 100.0,
        zeta_D: float = 1.0,
        n_pilots: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.encoder = encoder.eval()
        self.S_theta_D = S_theta_D.eval()
        self.Nr, self.Nt, self.K, self.T, self.Nu = Nr, Nt, K, T, Nu
        self.J = J
        self.zeta_D = zeta_D
        self.n_pilots = n_pilots if n_pilots is not None else 2 * Nt
        self.device = device

        self.sigmas_D = noise_schedule_exponential(sigma_D_1, sigma_D_J, J, device)

        for net in [self.encoder, self.S_theta_D]:
            for p in net.parameters():
                p.requires_grad_(False)

    def _generate_pilots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        NtK = self.Nt * self.K
        T_p = self.n_pilots
        dft = torch.zeros(NtK, T_p, dtype=torch.complex64, device=device)
        for i in range(NtK):
            for j in range(T_p):
                dft[i, j] = torch.exp(torch.tensor(-2j * math.pi * i * j / NtK))
        dft = dft / math.sqrt(NtK)
        return dft.unsqueeze(0).expand(batch_size, -1, -1)

    def run(
        self,
        D0_shape: Tuple,
        H0: torch.Tensor,
        Y: torch.Tensor,
        sigma_n: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run DPS-MIMO inference.

        Args:
            D0_shape: Shape of source image batch (B, 3, 256, 256).
            H0:       (B, Nu, NrK, NtK) true channel (for pilot estimation).
            Y:        (B, NrK, T) received signal.
            sigma_n:  Noise std.

        Returns:
            D_hat: Reconstructed image.
            H_hat: Estimated channel from pilots.
        """
        B = D0_shape[0]
        NrK = self.Nr * self.K
        device = Y.device
        scale = sigma_n / math.sqrt(2)

        # Pilot-based channel estimation
        X_pilot = self._generate_pilots(B, device)
        Y_pilot = sum(torch.bmm(H0[:, i], X_pilot) for i in range(self.Nu))
        Y_pilot = Y_pilot + torch.complex(
            torch.randn_like(Y_pilot.real) * scale,
            torch.randn_like(Y_pilot.imag) * scale,
        )
        H_hat = lmmse_channel_estimate(Y_pilot, X_pilot, sigma_n, self.Nr, self.Nt, self.K)

        # DPS image reconstruction
        D_j = torch.randn(B, 3, 256, 256, device=device) * self.sigmas_D[-1].item()

        for j in range(self.J, 0, -1):
            sigma_j = self.sigmas_D[j].item()
            sigma_j1 = self.sigmas_D[j - 1].item()
            eps_D = self.zeta_D * abs(sigma_j1 ** 2 - sigma_j ** 2)

            # Prior score
            sigma_vec = torch.full((B,), sigma_j, device=device)
            with torch.no_grad():
                score_prior = self.S_theta_D(D_j, sigma_vec)
                D_hat_tweedie = D_j + sigma_j ** 2 * score_prior

            # Likelihood gradient via DPS
            D_j_grad = D_j.detach().requires_grad_(True)
            sigma_vec_g = torch.full((B,), sigma_j, device=device)
            score_g = self.S_theta_D(D_j_grad, sigma_vec_g)
            D_hat_g = D_j_grad + sigma_j ** 2 * score_g
            X_hat = self.encoder(D_hat_g)
            if X_hat.dim() == 4:
                X_hat = X_hat[:, 0]
            HX = torch.bmm(H_hat, X_hat)
            residual = Y - HX
            lik_loss = (residual.abs() ** 2).sum() / (sigma_n ** 2)
            lik_grad = torch.autograd.grad(lik_loss, D_j_grad)[0]

            noise = torch.randn_like(D_j) * math.sqrt(eps_D)
            D_j = D_j + eps_D * score_prior - lik_grad * eps_D + noise

        D_hat = D_j.clamp(-1, 1)
        return D_hat, H_hat
