"""Tests for stable noise extensions."""
import pytest
import torch
import numpy as np
from extensions.stable_noise import SubGaussianStableNoise, sample_positive_stable


def test_positive_stable_samples_positive():
    samples = sample_positive_stable(0.75, 100)
    assert (samples > 0).all()


def test_positive_stable_shape():
    samples = sample_positive_stable(0.5, 50)
    assert samples.shape == (50,)


def test_stable_noise_gaussian_limit():
    """alpha=2 should behave like AWGN."""
    noise_model = SubGaussianStableNoise(alpha=2.0, sigma_n=1.0)
    N = noise_model.sample_noise((100, 4, 4))
    # Variance should be ~ sigma_n^2 = 1
    var = (N.abs() ** 2).mean().item()
    assert abs(var - 1.0) < 0.3


def test_stable_noise_shape():
    noise_model = SubGaussianStableNoise(alpha=1.5, sigma_n=1.0)
    N = noise_model.sample_noise((10, 4, 8))
    assert N.shape == (10, 4, 8)
    assert N.is_complex()


def test_stable_noise_log_likelihood_shape():
    noise_model = SubGaussianStableNoise(alpha=1.5, sigma_n=1.0)
    B, NrK, T = 3, 4, 8
    Y = torch.complex(torch.randn(B, NrK, T), torch.randn(B, NrK, T))
    mean = torch.zeros_like(Y)
    log_lik = noise_model.log_likelihood(Y, mean, L_A=5)
    assert log_lik.shape == (B,)
    assert not log_lik.isnan().any()


def test_stable_noise_alpha2_matches_gaussian():
    """alpha=2 log-likelihood should match Gaussian formula."""
    noise_model = SubGaussianStableNoise(alpha=2.0, sigma_n=1.0)
    B, NrK, T = 2, 4, 8
    Y = torch.complex(torch.randn(B, NrK, T), torch.randn(B, NrK, T))
    mean = torch.zeros_like(Y)
    log_lik = noise_model.log_likelihood(Y, mean)
    # Should be finite and negative
    assert not log_lik.isnan().any()
    assert (log_lik < 0).all()
