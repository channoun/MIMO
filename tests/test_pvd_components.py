"""Tests for PVD core components (tweedie, second_order, likelihood)."""
import pytest
import torch
import torch.nn as nn
from pvd.tweedie import tweedie_channel, tweedie_image
from pvd.second_order import compute_trace_score_channel, compute_trace_score_image
from pvd.likelihood import likelihood_score


class MockChannelScoreNet(nn.Module):
    """Returns a fixed score (zeros) for testing."""
    def forward(self, H, sigma):
        return torch.zeros_like(H)


class MockImageScoreNet(nn.Module):
    def forward(self, D, sigma):
        return torch.zeros_like(D)


class MockEncoder(nn.Module):
    """Returns a fixed output for testing."""
    def __init__(self, NtK=4, T=4):
        super().__init__()
        self.NtK = NtK
        self.T = T
        self.proj = nn.Linear(3 * 256 * 256, 2 * NtK * T)

    def forward(self, D):
        B = D.shape[0]
        x = D.view(B, -1)
        out = self.proj(x)
        real = out[:, :self.NtK * self.T].view(B, 1, self.NtK, self.T)
        imag = out[:, self.NtK * self.T:].view(B, 1, self.NtK, self.T)
        return torch.complex(real, imag)


def test_tweedie_channel_shape():
    B, NrK, NtK = 2, 8, 4
    H_j = torch.complex(torch.randn(B, NrK, NtK), torch.randn(B, NrK, NtK))
    net = MockChannelScoreNet()
    H_hat = tweedie_channel(H_j, 0.5, net)
    assert H_hat.shape == (B, NrK, NtK)
    assert H_hat.is_complex()


def test_tweedie_image_shape():
    B = 2
    D_j = torch.randn(B, 3, 256, 256)
    net = MockImageScoreNet()
    D_hat = tweedie_image(D_j, 0.5, net)
    assert D_hat.shape == (B, 3, 256, 256)


def test_tweedie_zero_score():
    """With zero score, Tweedie estimate = input."""
    B, NrK, NtK = 2, 4, 2
    H_j = torch.complex(torch.randn(B, NrK, NtK), torch.randn(B, NrK, NtK))
    net = MockChannelScoreNet()
    H_hat = tweedie_channel(H_j, 0.5, net)
    assert torch.allclose(H_hat.real, H_j.real)
    assert torch.allclose(H_hat.imag, H_j.imag)


def test_likelihood_score_shape():
    B, NrK, NtK, T = 1, 4, 2, 4
    H_j = torch.complex(torch.randn(B, NrK, NtK), torch.randn(B, NrK, NtK))
    D_j = torch.randn(B, 3, 256, 256)
    Y = torch.complex(torch.randn(B, NrK, T), torch.randn(B, NrK, T))
    S_H = MockChannelScoreNet()
    S_D = MockImageScoreNet()
    enc = MockEncoder(NtK=NtK, T=T)
    eff_var = torch.ones(B) * 0.1
    gH, gD = likelihood_score(H_j, D_j, Y, enc, S_H, S_D, 0.5, 0.5, eff_var, use_checkpoint=False)
    assert gH.shape == H_j.shape
    assert gD.shape == D_j.shape
