"""Unit tests for the quantization tables."""

from __future__ import annotations

import numpy as np

from dct_core.quant import LUM_TABLE_Q50, dequantize, quantize, scale_table


def test_q100_means_unit_step():
    t = scale_table(LUM_TABLE_Q50, 100)
    assert np.all(t == 1)


def test_q1_is_aggressive():
    t = scale_table(LUM_TABLE_Q50, 1)
    assert np.any(t >= 64)


def test_q50_is_base_table():
    t = scale_table(LUM_TABLE_Q50, 50)
    np.testing.assert_array_equal(t, LUM_TABLE_Q50)


def test_quantize_dequantize_error_bound():
    rng = np.random.default_rng(42)
    coeffs = rng.standard_normal((8, 8)) * 50.0
    q = quantize(coeffs, LUM_TABLE_Q50)
    rec = dequantize(q, LUM_TABLE_Q50)
    err = np.abs(coeffs - rec)
    # Per-element quantization error must be bounded by half the step.
    assert np.all(err <= LUM_TABLE_Q50 / 2.0 + 1e-9)


def test_invalid_q_raises():
    try:
        scale_table(LUM_TABLE_Q50, 0)
    except ValueError:
        return
    raise AssertionError("scale_table should reject Q outside [1, 100]")
