"""
dct_core.quant
==============
JPEG-style quantization for DCT coefficients.

We hard-code the standard luminance table used by JPEG at quality factor 50
and the standard scaling rule so the same table can be adapted to any Q
between 1 and 100. Higher Q = finer quantization = bigger file = better quality.

This is the heart of why DCT compression is *lossy*: dividing by the
quantization table and rounding to an integer throws away small
high-frequency details that the eye barely notices.
"""

from __future__ import annotations

import numpy as np


LUM_TABLE_Q50: np.ndarray = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 103, 77, 62],
        [30, 35, 55, 64, 81, 104, 92, 75],
        [50, 72, 78, 87, 103, 121, 100, 78],
        [68, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.int32,
)


def scale_table(base: np.ndarray, q: int) -> np.ndarray:
    """Scale a base quantization table for an arbitrary JPEG Q-factor in [1, 100].

    The rule is the standard one used by libjpeg:
        if Q < 50:  scale = 5000 / Q
        else:       scale = 200 - 2 * Q
        scaled[i]  = clip(floor((base[i] * scale + 50) / 100), 1, 255)
    """
    if not 1 <= q <= 100:
        raise ValueError(f"Quality factor Q must be in [1, 100], got {q}")
    scale = 5000 / q if q < 50 else 200 - 2 * q
    scaled = np.floor((base.astype(np.float64) * scale + 50.0) / 100.0)
    return np.clip(scaled, 1, 255).astype(np.int32)


def quantize(coeffs: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Quantize DCT coefficients: round(C / table). Accepts (8, 8) or (N, 8, 8)."""
    if coeffs.shape[-2:] != table.shape:
        raise ValueError(
            f"quantize: coefficient shape {coeffs.shape} does not match table {table.shape}"
        )
    return np.round(coeffs.astype(np.float64) / table.astype(np.float64)).astype(np.int32)


def dequantize(qcoeffs: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Dequantize: qcoeffs * table (recovers approximate DCT coefficients)."""
    if qcoeffs.shape[-2:] != table.shape:
        raise ValueError(
            f"dequantize: qcoeff shape {qcoeffs.shape} does not match table {table.shape}"
        )
    return qcoeffs.astype(np.float64) * table.astype(np.float64)
