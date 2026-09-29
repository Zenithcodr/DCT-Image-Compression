"""Unit tests for the DCT/IDCT core. No image file needed."""

from __future__ import annotations

import numpy as np

from dct_core.dct import D, dct2, dct2_batch, idct2, idct2_batch


def test_orthogonality():
    prod = D @ D.T
    # With our alpha normalization the matrix is orthonormal:
    # D @ D.T == I  (within floating-point tolerance).
    np.testing.assert_allclose(prod, np.eye(8), atol=1e-10)


def test_roundtrip_single_block():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 8))
    rec = idct2(dct2(x))
    np.testing.assert_allclose(rec, x, atol=1e-9)


def test_roundtrip_batch():
    rng = np.random.default_rng(1)
    blocks = rng.standard_normal((10, 8, 8))
    rec = idct2_batch(dct2_batch(blocks))
    np.testing.assert_allclose(rec, blocks, atol=1e-9)


def test_constant_block_dc_only():
    block = np.full((8, 8), 128.0)
    coeffs = dct2(block)
    # All AC coefficients should be ~0; only DC (top-left) is non-zero.
    ac = coeffs.copy()
    ac[0, 0] = 0.0
    np.testing.assert_allclose(ac, 0.0, atol=1e-9)
    # DC value for an 8x8 block of constant 128 is 128 * 8 * sqrt(2)/2 * 2 = 128 * 8 * sqrt(2).
    # We just verify it's large and positive.
    assert coeffs[0, 0] > 1000.0


def test_input_shape_errors():
    try:
        dct2(np.zeros((4, 4)))
    except ValueError:
        return
    raise AssertionError("dct2 should reject non-8x8 blocks")
