"""
BPG + 5G LDPC baseline with pilot-based channel estimation.

Traditional digital pipeline:
  1. BPG image compression (uses subprocess call to bpgenc/bpgdec).
  2. 5G LDPC channel coding (simulated).
  3. Pilot-based LMMSE channel estimation.
  4. QPSK/QAM modulation + ZF equalization.

Note: BPG encoder/decoder binaries must be installed separately.
This module provides a simulation-based approximation when BPG is unavailable.

Reference: [9] in the paper.
"""
import os
import math
import tempfile
import subprocess
import warnings
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from PIL import Image

from channels.rayleigh import lmmse_channel_estimate


def _bpg_available() -> bool:
    try:
        subprocess.run(["bpgenc", "--help"], capture_output=True, timeout=2)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _compress_bpg(img_np: np.ndarray, quality: int = 30) -> bytes:
    """Compress image with BPG and return bytes."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_in:
        in_path = f_in.name
    with tempfile.NamedTemporaryFile(suffix=".bpg", delete=False) as f_out:
        out_path = f_out.name
    try:
        Image.fromarray(img_np).save(in_path)
        subprocess.run(["bpgenc", "-q", str(quality), "-o", out_path, in_path],
                       check=True, capture_output=True)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def _decompress_bpg(bpg_bytes: bytes) -> np.ndarray:
    """Decompress BPG bytes to numpy image."""
    with tempfile.NamedTemporaryFile(suffix=".bpg", delete=False) as f_in:
        f_in.write(bpg_bytes)
        in_path = f_in.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_out:
        out_path = f_out.name
    try:
        subprocess.run(["bpgdec", "-o", out_path, in_path],
                       check=True, capture_output=True)
        return np.array(Image.open(out_path).convert("RGB"))
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


class BPGLDPCBaseline:
    """
    BPG + 5G LDPC + pilot-based channel estimation baseline.

    When BPG binaries are unavailable, falls back to JPEG compression
    at an equivalent quality level (for testing purposes only).

    Args:
        Nr, Nt, K, T: Channel dimensions.
        Nu:           Number of users.
        bpg_quality:  BPG quality parameter (lower = more compression).
        cbr_target:   Target channel bandwidth ratio.
        n_pilots:     Number of pilot symbols.
    """

    def __init__(
        self,
        Nr: int,
        Nt: int,
        K: int,
        T: int,
        Nu: int = 1,
        bpg_quality: int = 30,
        cbr_target: float = 1/6,
        n_pilots: Optional[int] = None,
    ):
        self.Nr, self.Nt, self.K, self.T, self.Nu = Nr, Nt, K, T, Nu
        self.bpg_quality = bpg_quality
        self.cbr_target = cbr_target
        self.n_pilots = n_pilots if n_pilots is not None else 2 * Nt
        self._bpg_ok = _bpg_available()
        if not self._bpg_ok:
            warnings.warn(
                "BPG encoder/decoder not found. Using JPEG fallback. "
                "Install bpgenc/bpgdec for accurate BPG-LDPC baseline.",
                UserWarning,
            )

    def _compress(self, img_tensor: torch.Tensor) -> Tuple[bytes, float]:
        """Compress a single image tensor [3, H, W] in [-1, 1] to bytes."""
        img_np = ((img_tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
        if self._bpg_ok:
            data = _compress_bpg(img_np, self.bpg_quality)
        else:
            # JPEG fallback
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                path = f.name
            Image.fromarray(img_np).save(path, quality=self.bpg_quality)
            with open(path, "rb") as f:
                data = f.read()
            os.unlink(path)
        bpp = len(data) * 8 / (img_tensor.shape[1] * img_tensor.shape[2])
        return data, bpp

    def _decompress(self, data: bytes) -> torch.Tensor:
        """Decompress bytes to image tensor [3, H, W] in [-1, 1]."""
        if self._bpg_ok:
            img_np = _decompress_bpg(data)
        else:
            import io
            img_np = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 127.5 - 1.0
        return img_t

    def _simulate_ldpc_awgn(
        self,
        data_bits: bytes,
        snr_db: float,
        cbr: float,
    ) -> bytes:
        """
        Simulate 5G LDPC + QAM transmission over estimated AWGN channel.

        This is a simplified simulation: at high SNR we return data unchanged;
        at low SNR we flip bits with probability corresponding to LDPC BER.
        In practice, use a real LDPC library (e.g., aff3ct).
        """
        snr_linear = 10 ** (snr_db / 10.0)
        # Approximate BER for LDPC-QPSK at this SNR
        ber = 0.5 * math.erfc(math.sqrt(snr_linear))
        if ber < 1e-5:
            return data_bits  # Essentially error-free
        # Flip bits randomly (simplified)
        bit_array = np.frombuffer(data_bits, dtype=np.uint8).copy()
        mask = np.random.random(len(bit_array)) < ber * 8
        bit_array[mask] ^= np.random.randint(1, 256, mask.sum(), dtype=np.uint8)
        return bit_array.tobytes()

    def run(
        self,
        D0: torch.Tensor,
        H0: torch.Tensor,
        snr_db: float,
        sigma_n: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run BPG-LDPC baseline.

        Returns:
            D_hat: (B, 3, 256, 256) reconstructed images.
            H_hat: (B, NrK, NtK) estimated channels.
        """
        B = D0.shape[0]
        device = D0.device
        NrK = self.Nr * self.K
        NtK = self.Nt * self.K
        scale = sigma_n / math.sqrt(2)

        D_hats = []
        for b in range(B):
            compressed, bpp = self._compress(D0[b])
            # Simulate channel transmission
            noisy = self._simulate_ldpc_awgn(compressed, snr_db, self.cbr_target)
            try:
                D_hat_b = self._decompress(noisy)
            except Exception:
                # Decoding failure: return zeros
                D_hat_b = torch.zeros(3, 256, 256)
            D_hats.append(D_hat_b)
        D_hat = torch.stack(D_hats, dim=0).to(device)

        # Pilot-based channel estimation
        X_pilot = torch.zeros(B, NtK, self.n_pilots, dtype=torch.complex64, device=device)
        for i in range(NtK):
            for j in range(self.n_pilots):
                X_pilot[:, i, j] = torch.exp(
                    torch.tensor(-2j * math.pi * i * j / NtK)
                ) / math.sqrt(NtK)

        Y_pilot = sum(torch.bmm(H0[:, i], X_pilot) for i in range(self.Nu))
        Y_pilot = Y_pilot + torch.complex(
            torch.randn_like(Y_pilot.real) * scale,
            torch.randn_like(Y_pilot.imag) * scale,
        )
        H_hat = lmmse_channel_estimate(Y_pilot, X_pilot, sigma_n, self.Nr, self.Nt, self.K)

        return D_hat, H_hat
