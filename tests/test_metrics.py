"""Unit tests for the quality metrics."""

from __future__ import annotations

import numpy as np

from quality.metrics import (
    compression_ratio,
    estimate_compressed_size,
    mse,
    psnr,
)


def test_mse_zero_for_identical():
    a = np.random.default_rng(0).integers(0, 256, (16, 16), dtype=np.uint8)
    assert mse(a, a) == 0.0


def test_psnr_inf_for_identical():
    a = np.random.default_rng(1).integers(0, 256, (16, 16), dtype=np.uint8)
    assert psnr(a, a) == float("inf")


def test_psnr_known_noise():
    rng = np.random.default_rng(2)
    a = np.zeros((64, 64), dtype=np.float64)
    b = rng.normal(0.0, 10.0, (64, 64))
    # MSE of zero-mean Gaussian noise with sigma=10 is sigma^2 = 100.
    assert 95.0 < mse(a, b) < 105.0
    # PSNR = 10 * log10(255^2 / 100) ≈ 28.13 dB.
    p = psnr(a, b)
    assert 26.0 < p < 30.0


def test_compression_ratio_basic():
    assert compression_ratio(1000, 250) == 4.0


def test_estimate_size_responds_to_q():
    # Fake "quantized" blocks: a few zeros, a few small ints.
    base = np.zeros((4, 8, 8), dtype=np.int32)
    base[0, 0, 0] = 10
    base[1, 0, 1] = -5
    size_low = estimate_compressed_size(base)
    dense = np.full((4, 8, 8), 10, dtype=np.int32)
    size_high = estimate_compressed_size(dense)
    assert size_high > size_low
