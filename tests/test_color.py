"""Tests for the RGB / YCbCr colour pipeline."""

from __future__ import annotations

import numpy as np

from pipeline import compress_image
from pipeline.color import rgb_to_ycbcr, ycbcr_to_rgb


def test_roundtrip_rgb():
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    ycbcr = rgb_to_ycbcr(rgb)
    rec = ycbcr_to_rgb(ycbcr)
    # Forward and inverse use 4-decimal BT.601 coefficients, so the round-trip
    # is at most ±1 in any channel due to fp rounding. That's the expected
    # behaviour for any float-based colour converter.
    diff = np.abs(rec.astype(np.int16) - rgb.astype(np.int16))
    assert int(diff.max()) <= 1


def test_compress_color_image_runs():
    rng = np.random.default_rng(4)
    rgb = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    r = compress_image(rgb, quality=75)
    assert r.reconstructed.shape == rgb.shape
    assert r.quality == 75
    assert r.chroma_quality == 75 // 2
    # Random noise is incompressible; MSE is allowed to be large, just finite.
    assert 0.0 <= r.mse_val < 10000.0
    assert r.psnr_val >= 0.0
    assert r.cr > 0.0


def test_color_size_includes_three_channels():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    r = compress_image(rgb, quality=75)
    assert r.original_bytes == 64 * 64 * 3
