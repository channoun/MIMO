"""
DJSCC-MIMO baseline with pilot-based LMMSE channel estimation.

Two variants:
  1. Perfect CSI:   Oracle upper bound — H0 is known exactly.
  2. Pilots:        2*Nt pilot symbols, LMMSE estimation, then DJSCC decoding.

Reference: Wu et al., "MIMO-JSCC" [26] in the paper.
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from channels.rayleigh import lmmse_channel_estimate


class DJSCCMIMOBaseline:
    """
    DJSCC-MIMO baseline (pilot-based or perfect CSI).

    At inference:
      1. Estimate H using pilots (or use perfect H0).
      2. Compute X_hat = pseudo-inverse(H_hat) @ Y  (zero-forcing equalization).
      3. Decode image from X_hat using DJSCC decoder.

    Args:
        encoder:    Trained DJSCCEncoder f_γ.
        decoder:    Trained DJSCCDecoder g_β.
        Nr, Nt, K, T: Channel dimensions.
        Nu:         Number of users.
        perfect_csi: If True, use ground-truth H0 (oracle bound).
        n_pilots:   Number of pilot columns (default 2*Nt per block).
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
        perfect_csi: bool = False,
        n_pilots: Optional[int] = None,
    ):
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()
        self.Nr, self.Nt, self.K, self.T, self.Nu = Nr, Nt, K, T, Nu
        self.perfect_csi = perfect_csi
        self.n_pilots = n_pilots if n_pilots is not None else 2 * Nt

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _generate_pilots(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate orthogonal pilot matrix X_pilot ∈ C^{NtK × n_pilots}.
        Uses a DFT-based construction (optimal for LMMSE under i.i.d. Rayleigh).
        """
        NtK = self.Nt * self.K
        T_p = self.n_pilots
        # DFT matrix scaled by sqrt(NtK) for unit power per element
        dft = torch.zeros(NtK, T_p, dtype=torch.complex64, device=device)
        for i in range(NtK):
            for j in range(T_p):
                dft[i, j] = torch.exp(
                    torch.tensor(-2j * math.pi * i * j / NtK)
                )
        dft = dft / math.sqrt(NtK)
        return dft.unsqueeze(0).expand(batch_size, -1, -1)  # (B, NtK, T_p)

    @torch.no_grad()
    def _zero_forcing(
        self,
        H_hat: torch.Tensor,
        Y_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero-forcing equalization: X_hat = (H_hat^H H_hat)^{-1} H_hat^H Y_data

        Args:
            H_hat:   (B, NrK, NtK) complex estimated channel.
            Y_data:  (B, NrK, T) complex received signal.

        Returns:
            X_hat: (B, NtK, T) equalized signal.
        """
        # Pseudo-inverse via least-squares solve
        # min_X ||Y - H X||^2  →  X = (H^H H)^{-1} H^H Y
        HH = torch.bmm(H_hat.conj().transpose(-2, -1), H_hat)  # (B, NtK, NtK)
        HY = torch.bmm(H_hat.conj().transpose(-2, -1), Y_data)  # (B, NtK, T)
        X_hat = torch.linalg.solve(HH, HY)  # (B, NtK, T)
        return X_hat

    @torch.no_grad()
    def run(
        self,
        D0: torch.Tensor,
        H0: torch.Tensor,
        snr_db: float,
        sigma_n: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run baseline inference.

        Args:
            D0:      (B, 3, 256, 256) source images.
            H0:      (B, Nu, NrK, NtK) true channel.
            snr_db:  SNR in dB (for documentation only; sigma_n used for estimation).
            sigma_n: Noise standard deviation.

        Returns:
            D_hat: (B, 3, 256, 256) reconstructed images.
            H_hat: (B, NrK, NtK) estimated channel.
        """
        B = D0.shape[0]
        device = D0.device

        # Encode
        X = self.encoder(D0)  # (B, Nu, NtK, T)

        # Channel application (simulate reception) — we reconstruct Y from H0 and X
        NrK = self.Nr * self.K
        HX = sum(torch.bmm(H0[:, i], X[:, i]) for i in range(self.Nu))
        scale = sigma_n / math.sqrt(2)
        noise = torch.complex(
            torch.randn(B, NrK, self.T, device=device) * scale,
            torch.randn(B, NrK, self.T, device=device) * scale,
        )
        Y = HX + noise

        # Channel estimation
        if self.perfect_csi:
            H_hat = H0[:, 0]  # (B, NrK, NtK) — use first user for single-user
        else:
            X_pilot = self._generate_pilots(B, device)
            Y_pilot_signal = sum(
                torch.bmm(H0[:, i], X_pilot) for i in range(self.Nu)
            )
            pilot_noise = torch.complex(
                torch.randn(B, NrK, self.n_pilots, device=device) * scale,
                torch.randn(B, NrK, self.n_pilots, device=device) * scale,
            )
            Y_pilot = Y_pilot_signal + pilot_noise
            H_hat = lmmse_channel_estimate(
                Y_pilot, X_pilot, sigma_n, self.Nr, self.Nt, self.K
            )

        # Equalization
        X_hat = self._zero_forcing(H_hat, Y)  # (B, NtK, T)

        # Decode
        D_hat = self.decoder(X_hat)

        return D_hat, H_hat
