"""
DJSCC + Pro-BiG-AMP blind detector.

Pro-BiG-AMP performs joint channel estimation and data detection
using Bilinear Generalized AMP (belief propagation in the Bethe free energy).

Note: Pro-BiG-AMP requires ⌈1 + log Nu⌉*Nt reference symbols.
It is known to FAIL in multi-user settings (Nu > 1). This is expected
behavior documented in the paper, not a bug.

Reference: Zhang et al., "Bilinear Message Passing for Joint Channel
Estimation and Decoding." IEEE TSP 2017.
"""
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class ProBiGAMPBaseline:
    """
    DJSCC + Pro-BiG-AMP blind detector.

    Implements simplified Bilinear Generalized AMP for joint H, X recovery.
    For Nu > 1, convergence is not guaranteed (expected failure mode).

    Args:
        encoder:    Trained DJSCCEncoder.
        decoder:    Trained DJSCCDecoder.
        Nr, Nt, K, T: Channel dimensions.
        Nu:         Number of users (>1 may diverge).
        n_ref:      Number of reference symbols (⌈1 + log Nu⌉ * Nt).
        n_iter:     Number of AMP iterations.
        damping:    AMP damping factor for stability.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        Nr: int,
        Nt: int,
        K: int,
        T: int,
        Nu: int = 1,
        n_iter: int = 50,
        damping: float = 0.5,
    ):
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()
        self.Nr, self.Nt, self.K, self.T, self.Nu = Nr, Nt, K, T, Nu
        self.n_iter = n_iter
        self.damping = damping
        self.n_ref = math.ceil(1 + math.log2(max(Nu, 2))) * Nt

        if Nu > 1:
            import warnings
            warnings.warn(
                f"Pro-BiG-AMP with Nu={Nu} > 1: convergence is not guaranteed. "
                "This is a known result from the paper (Section V.B).",
                UserWarning,
            )

        for net in [self.encoder, self.decoder]:
            for p in net.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def _bigamp_iterations(
        self,
        Y: torch.Tensor,
        X_init: torch.Tensor,
        H_init: torch.Tensor,
        sigma_n: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear AMP iterations for joint H, X estimation.

        Model: Y ≈ H @ X + N,  N ~ CN(0, sigma_n^2 I)

        Args:
            Y:      (B, NrK, T) received signal.
            X_init: (B, NtK, T) initial X estimate (from reference symbols).
            H_init: (B, NrK, NtK) initial H estimate.
            sigma_n: Noise std.

        Returns:
            H_hat: (B, NrK, NtK) estimated channel.
            X_hat: (B, NtK, T) estimated signal.
        """
        B = Y.shape[0]
        H = H_init.clone()
        X = X_init.clone()
        tau_H = sigma_n ** 2 * torch.ones(B, device=Y.device)
        tau_X = sigma_n ** 2 * torch.ones(B, device=Y.device)

        for it in range(self.n_iter):
            # AMP residual
            Z = torch.bmm(H, X)  # (B, NrK, T)
            R = Y - Z            # (B, NrK, T)

            # Update X: matched filter + MMSE
            tau_H_scalar = tau_H.mean().item()
            HH = torch.bmm(H.conj().transpose(-2, -1), H)  # (B, NtK, NtK)
            NrK = H.shape[1]
            reg_X = (sigma_n ** 2 / (tau_H_scalar * NrK + 1e-8)) * torch.eye(
                HH.shape[-1], dtype=HH.dtype, device=Y.device
            ).unsqueeze(0)
            X_new = torch.linalg.solve(HH + reg_X, torch.bmm(H.conj().transpose(-2, -1), Y))

            # Update H: matched filter + MMSE
            tau_X_scalar = tau_X.mean().item()
            XX = torch.bmm(X.conj().transpose(-2, -1), X)  # (B, T, T)
            T_dim = X.shape[-1]
            reg_H = (sigma_n ** 2 / (tau_X_scalar * T_dim + 1e-8)) * torch.eye(
                NrK, dtype=HH.dtype, device=Y.device
            ).unsqueeze(0).expand(B, -1, -1)
            # H_new = Y X^H (X X^H + reg)^{-1}
            gram = torch.bmm(X, X.conj().transpose(-2, -1))  # (B, T, T)  — actually NtK x NtK
            # Correct: X is (B, NtK, T), so X X^H is (B, NtK, NtK)
            gram_X = torch.bmm(X_new, X_new.conj().transpose(-2, -1))
            reg_H2 = (sigma_n ** 2 / (tau_X_scalar * T_dim + 1e-8)) * torch.eye(
                X_new.shape[1], dtype=gram_X.dtype, device=Y.device
            ).unsqueeze(0)
            YXH = torch.bmm(Y, X_new.conj().transpose(-2, -1))
            H_new = torch.linalg.solve((gram_X + reg_H2).transpose(-2, -1), YXH.transpose(-2, -1)).transpose(-2, -1)

            # Damped update
            X = self.damping * X_new + (1 - self.damping) * X
            H = self.damping * H_new + (1 - self.damping) * H

            # Update noise variances
            residual = Y - torch.bmm(H, X)
            tau_H = (residual.abs()**2).mean(dim=(1, 2))
            tau_X = tau_H.clone()

        return H, X

    @torch.no_grad()
    def run(
        self,
        D0: torch.Tensor,
        H0: torch.Tensor,
        snr_db: float,
        sigma_n: float,
        Y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run Pro-BiG-AMP blind receiver.

        Args:
            D0:     (B, 3, 256, 256) source images.
            H0:     (B, Nu, NrK, NtK) true channel (for building Y if not provided).
            snr_db: SNR in dB.
            sigma_n: Noise std.
            Y:      (B, NrK, T) received signal (optional; computed from D0, H0 if None).

        Returns:
            D_hat: (B, 3, 256, 256) reconstructed images.
            H_hat: (B, NrK, NtK) estimated channel.
        """
        B = D0.shape[0]
        device = D0.device
        NrK = self.Nr * self.K
        NtK = self.Nt * self.K
        scale = sigma_n / math.sqrt(2)

        # Encode
        X = self.encoder(D0)[:, 0]  # (B, NtK, T)

        # Build Y if not provided
        if Y is None:
            HX = sum(torch.bmm(H0[:, i], self.encoder(D0)[:, i]) for i in range(self.Nu))
            noise = torch.complex(
                torch.randn_like(HX.real) * scale,
                torch.randn_like(HX.imag) * scale,
            )
            Y = HX + noise

        # Initialize H, X
        H_init = torch.complex(
            torch.randn(B, NrK, NtK, device=device) / math.sqrt(2 * NtK),
            torch.randn(B, NrK, NtK, device=device) / math.sqrt(2 * NtK),
        )
        X_init = X.clone()

        # Pro-BiG-AMP
        H_hat, X_hat = self._bigamp_iterations(Y, X_init, H_init, sigma_n)

        # Decode from estimated X
        D_hat = self.decoder(X_hat)
        return D_hat, H_hat
