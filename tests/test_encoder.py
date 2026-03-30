"""Tests for encoder module."""
import pytest
import torch
from encoder.swin_jscc import DJSCCEncoder, DJSCCDecoder


@pytest.fixture
def small_encoder():
    """Small encoder for fast testing."""
    return DJSCCEncoder(
        embed_dim=32,
        depths=[1, 1, 2, 1],
        num_heads=[1, 2, 4, 8],
        window_size=4,
        Nt=1, K=4, T=4, Nu=1,
    )


@pytest.fixture
def small_decoder():
    return DJSCCDecoder(
        embed_dim=32,
        depths=[1, 2, 1, 1],
        num_heads=[8, 4, 2, 1],
        window_size=4,
        Nt=1, K=4, T=4,
    )


def test_encoder_output_shape(small_encoder):
    D0 = torch.randn(2, 3, 256, 256)
    X = small_encoder(D0)
    assert X.shape == (2, 1, 4, 4)  # (B, Nu, NtK, T)
    assert X.is_complex()


def test_encoder_power_constraint(small_encoder):
    """Power constraint: E[|X|^2] should be ~1."""
    D0 = torch.randn(4, 3, 256, 256)
    X = small_encoder(D0)
    power = (X.abs()**2).mean().item()
    assert abs(power - 1.0) < 0.1


def test_decoder_output_shape(small_encoder, small_decoder):
    D0 = torch.randn(2, 3, 256, 256)
    X = small_encoder(D0)
    D_hat = small_decoder(X)
    assert D_hat.shape == (2, 3, 256, 256)


def test_decoder_output_range(small_encoder, small_decoder):
    """Decoder output should be in [-1, 1] (Tanh activation)."""
    D0 = torch.randn(2, 3, 256, 256)
    X = small_encoder(D0)
    D_hat = small_decoder(X)
    assert D_hat.min().item() >= -1.0 - 1e-5
    assert D_hat.max().item() <= 1.0 + 1e-5


def test_encoder_gradient_flow(small_encoder, small_decoder):
    """Gradient should flow through encoder -> decoder."""
    D0 = torch.randn(1, 3, 256, 256, requires_grad=True)
    X = small_encoder(D0)
    D_hat = small_decoder(X)
    loss = D_hat.mean()
    loss.backward()
    assert D0.grad is not None
    assert not D0.grad.isnan().any()
