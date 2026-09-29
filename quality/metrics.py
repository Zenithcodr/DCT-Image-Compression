"""
quality.metrics
===============
Numerical measures of how different two images are.

  * MSE  (Mean Squared Error)         — average per-pixel squared difference.
  * PSNR (Peak Signal-to-Noise Ratio) — log-scale quality in decibels;
                                        larger PSNR means reconstruction is closer
                                        to the original. PSNR is defined only
                                        when MSE > 0, otherwise we report +inf.
  * Compression Ratio                 — original bytes / compressed bytes.
  * estimate_compressed_size          — runs a tiny zig-zag + run-length
                                        estimator over quantized blocks so the
                                        demo can show a *measured* compression
                                        ratio that responds to Q.
"""

from __future__ import annotations

import numpy as np


# Standard zig-zag scan order for an 8x8 block (indices 0..63).
_ZIGZAG: np.ndarray = np.array(
    [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ],
    dtype=np.int32,
)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error between two arrays of identical shape."""
    if a.shape != b.shape:
        raise ValueError(f"mse: shape mismatch {a.shape} vs {b.shape}")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio in decibels. Returns +inf if MSE == 0."""
    m = mse(a, b)
    if m <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_val * max_val) / m))


def estimate_compressed_size(quantized_blocks: np.ndarray) -> int:
    """Estimate the byte cost of the quantized DCT coefficients.

    For each 8x8 block we walk the zig-zag order, skip runs of zeros, and
    encode each non-zero as a 2-byte pair (run length, value). We also add
    1 byte for an end-of-block marker so the decoder can stop. This is a
    teaching estimator — not Huffman — but it does respond to Q (more
    aggressive Q → more zeros → fewer bytes).
    """
    if quantized_blocks.ndim != 3 or quantized_blocks.shape[1:] != (8, 8):
        raise ValueError(
            f"estimate_compressed_size expects (N, 8, 8), got {quantized_blocks.shape}"
        )
    flat = quantized_blocks.reshape(quantized_blocks.shape[0], 64)
    total = 0
    for row in flat:
        run = 0
        for k in _ZIGZAG:
            v = int(row[k])
            if v == 0:
                run += 1
                if run > 255:
                    run = 255
                continue
            total += 2
            run = 0
        total += 1
    return total


def compression_ratio(orig_bytes: int, comp_bytes: int) -> float:
    """orig_bytes / comp_bytes; safe when comp_bytes is 0."""
    if comp_bytes <= 0:
        return float("inf")
    return float(orig_bytes) / float(comp_bytes)
