"""Tests for metrics module."""
import pytest
import torch
from metrics.ms_ssim import ms_ssim, MSSSIM
from metrics.nmse import nmse_db, nmse_db_batch


def test_ms_ssim_identical():
    """MS-SSIM of identical images should be 1."""
    img = torch.rand(2, 3, 256, 256)
    vals = ms_ssim(img, img, data_range=1.0)
    assert vals.shape == (2,)
    assert (vals > 0.99).all()


def test_ms_ssim_different():
    """MS-SSIM of very different images should be small."""
    img1 = torch.zeros(2, 3, 256, 256)
    img2 = torch.ones(2, 3, 256, 256)
    vals = ms_ssim(img1, img2, data_range=1.0)
    assert (vals < 0.5).all()


def test_ms_ssim_range():
    """MS-SSIM should be in [0, 1]."""
    img1 = torch.rand(4, 3, 256, 256)
    img2 = torch.rand(4, 3, 256, 256)
    vals = ms_ssim(img1, img2, data_range=1.0)
    assert (vals >= 0).all()
    assert (vals <= 1.01).all()


def test_nmse_zero():
    """NMSE of perfect estimate should be very negative dB."""
    x = torch.randn(4, 64, 64)
    val = nmse_db(x, x)
    assert val.item() < -60.0


def test_nmse_positive():
    """NMSE of zeros estimate should be 0 dB."""
    x = torch.randn(4, 64, 64)
    val = nmse_db(torch.zeros_like(x), x)
    assert abs(val.item()) < 1.0  # ~0 dB


def test_nmse_complex():
    """NMSE works with complex tensors."""
    H = torch.complex(torch.randn(4, 8, 4), torch.randn(4, 8, 4))
    val = nmse_db(H, H)
    assert val.item() < -60.0


def test_nmse_batch():
    x = torch.randn(4, 64, 64)
    vals = nmse_db_batch(x, x)
    assert vals.shape == (4,)
    assert (vals < -60.0).all()


def test_msssim_module():
    module = MSSSIM(data_range=1.0)
    img1 = torch.rand(2, 3, 256, 256)
    img2 = torch.rand(2, 3, 256, 256)
    val = module(img1, img2)
    assert val.ndim == 0  # scalar
    assert 0 <= val.item() <= 1.01
