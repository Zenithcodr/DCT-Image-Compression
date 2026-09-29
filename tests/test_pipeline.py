"""End-to-end pipeline test using a synthetic image (no PNG needed)."""

from __future__ import annotations

import numpy as np

from dct_core.blocks import merge_blocks, split_blocks
from dct_core.dct import dct2_batch, idct2_batch
from dct_core.quant import LUM_TABLE_Q50, dequantize, quantize, scale_table
from quality.metrics import mse, psnr


def _compress(img: np.ndarray, q: int = 50) -> np.ndarray:
    shifted = img.astype(np.float64) - 128.0
    blocks, orig_shape, padded_shape = split_blocks(shifted)
    coeffs = dct2_batch(blocks)
    table = scale_table(LUM_TABLE_Q50, q)
    qblocks = quantize(coeffs, table)
    rec_coeffs = dequantize(qblocks, table)
    rec_shifted = idct2_batch(rec_coeffs)
    rec = merge_blocks(rec_shifted, orig_shape, padded_shape) + 128.0
    return np.clip(rec, 0, 255).astype(np.uint8)


def test_pipeline_smooth_image_high_psnr():
    rng = np.random.default_rng(7)
    # Smooth-ish image: low-frequency gradient + tiny noise.
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
    img = (0.5 * xx + 0.3 * yy + rng.normal(0, 1.0, (64, 64))).clip(0, 255).astype(np.uint8)
    rec = _compress(img, q=75)
    p = psnr(img, rec)
    assert p > 35.0, f"PSNR too low for smooth image: {p:.2f} dB"


def test_pipeline_non_multiple_of_8_dimensions():
    img = np.zeros((37, 53), dtype=np.uint8)
    rec = _compress(img, q=50)
    assert rec.shape == (37, 53)


def test_q100_is_lossless_within_one():
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    rec = _compress(img, q=100)
    assert rec.shape == img.shape
    assert int(np.max(np.abs(rec.astype(np.int16) - img.astype(np.int16)))) <= 1
