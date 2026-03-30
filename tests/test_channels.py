"""Tests for channels module."""
import math
import pytest
import torch
from channels.rayleigh import (
    generate_rayleigh_channel,
    apply_channel,
    apply_channel_with_noise,
    compute_snr,
    noise_schedule_exponential,
    lmmse_channel_estimate,
)


def test_rayleigh_channel_shape():
    H0 = generate_rayleigh_channel(4, 2, 4, 2, 3)
    assert H0.shape == (4, 2, 12, 6)
    assert H0.is_complex()


def test_rayleigh_channel_block_diagonal():
    """Off-diagonal blocks should be zero."""
    H0 = generate_rayleigh_channel(2, 1, 3, 2, 4)
    # Check that block (0,1) is zero
    assert H0[0, 0, :3, 2:4].abs().max() < 1e-6


def test_rayleigh_channel_unit_variance():
    """Each element should have variance ~1 (CN(0,1))."""
    H0 = generate_rayleigh_channel(1000, 1, 4, 2, 1)
    # Shape: (1000, 1, 4, 2), should have E[|h|^2]=1
    var = (H0.abs()**2).mean()
    assert abs(var.item() - 1.0) < 0.05


def test_apply_channel_shape():
    B, Nu, Nr, Nt, K, T = 2, 1, 4, 2, 3, 16
    H0 = generate_rayleigh_channel(B, Nu, Nr, Nt, K)
    X = torch.complex(
        torch.randn(B, Nu, Nt * K, T),
        torch.randn(B, Nu, Nt * K, T),
    )
    Y, sigma_n = apply_channel(H0, X, snr_db=10.0)
    assert Y.shape == (B, Nr * K, T)
    assert sigma_n > 0


def test_apply_channel_snr():
    """Empirical SNR should be close to target."""
    B, Nu, Nr, Nt, K, T = 100, 1, 4, 1, 1, 64
    H0 = generate_rayleigh_channel(B, Nu, Nr, Nt, K)
    # Normalize X to unit power per element so apply_channel's power=1.0 assumption holds
    X_raw = torch.complex(
        torch.randn(B, Nu, Nt * K, T),
        torch.randn(B, Nu, Nt * K, T),
    )
    pwr = (X_raw.abs()**2).mean(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    X = X_raw / pwr.sqrt()  # unit average power per element

    target_snr = 10.0
    _, sigma_n = apply_channel(H0, X, snr_db=target_snr)
    # Apply manually to check SNR
    Y, N = apply_channel_with_noise(H0, X, sigma_n)
    empirical_snr = compute_snr(H0, X, N)
    assert abs(empirical_snr - target_snr) < 5.0  # within 5 dB (statistical tolerance)


def test_noise_schedule_exponential():
    sigmas = noise_schedule_exponential(0.01, 100.0, 30, torch.device("cpu"))
    assert sigmas.shape == (31,)
    assert sigmas[0].item() == 0.0
    assert abs(sigmas[1].item() - 0.01) < 0.01
    assert abs(sigmas[-1].item() - 100.0) < 1.0
    # Should be monotonically increasing
    assert (sigmas[1:] > sigmas[:-1]).all()


def test_lmmse_shape():
    B, Nr, Nt, K, T_p = 2, 4, 2, 3, 8
    Y_pilot = torch.complex(torch.randn(B, Nr*K, T_p), torch.randn(B, Nr*K, T_p))
    X_pilot = torch.complex(torch.randn(B, Nt*K, T_p), torch.randn(B, Nt*K, T_p))
    H_hat = lmmse_channel_estimate(Y_pilot, X_pilot, 0.1, Nr, Nt, K)
    assert H_hat.shape == (B, Nr*K, Nt*K)
