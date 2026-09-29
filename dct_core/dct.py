"""
dct_core.dct
============
From-scratch 2-D Type-II Discrete Cosine Transform (DCT) and its inverse.

Why we implement it ourselves (instead of calling scipy.fft.dctn):
  * It keeps the math visible and aligned with the syllabus.
  * The 8x8 DCT matrix is precomputed once and reused for every block,
    so the cost on a 512x512 image is essentially 4096 tiny 8x8 matrix products.
  * No external FFT dependency to demo live.

The forward 2-D DCT of an 8x8 block x is:
    X = D @ x @ D.T
and the matching inverse (Type-III) is:
    x = D.T @ X @ D
where D is the 8x8 matrix whose (k, i) entry is:
    D[k, i] = alpha(k) * cos((2i + 1) * k * pi / 16)
with   alpha(0)     = 1/sqrt(8)
       alpha(k > 0) = sqrt(2/8) = 1/2.

With this normalization D is orthonormal, so D @ D.T == I and
D.T @ D == I. The inverse above is therefore the exact transpose.
"""

from __future__ import annotations

import numpy as np

BLOCK: int = 8


def _build_dct_matrix(n: int = 8, dtype=np.float64) -> np.ndarray:
    k = np.arange(n, dtype=dtype).reshape(n, 1)
    i = np.arange(n, dtype=dtype).reshape(1, n)
    alpha = np.where(k == 0, 1.0 / np.sqrt(n), np.sqrt(2.0 / n))
    return alpha * np.cos((2 * i + 1) * k * np.pi / (2 * n))


D: np.ndarray = _build_dct_matrix(BLOCK, dtype=np.float64)


def dct2(block: np.ndarray) -> np.ndarray:
    """Forward 2-D DCT-II of a single 8x8 block.

    Input shape (8, 8), float-like. Returns (8, 8) float64 coefficients.
    """
    if block.shape != (BLOCK, BLOCK):
        raise ValueError(f"dct2 expects an {(BLOCK, BLOCK)} block, got {block.shape}")
    x = np.asarray(block, dtype=np.float64)
    return D @ x @ D.T


def idct2(coeffs: np.ndarray) -> np.ndarray:
    """Inverse 2-D DCT-III (matching dct2) of a single 8x8 coefficient block."""
    if coeffs.shape != (BLOCK, BLOCK):
        raise ValueError(f"idct2 expects an {(BLOCK, BLOCK)} block, got {coeffs.shape}")
    X = np.asarray(coeffs, dtype=np.float64)
    return D.T @ X @ D


def dct2_batch(blocks: np.ndarray) -> np.ndarray:
    """Forward DCT applied to a stack of N blocks of shape (N, 8, 8).

    Computes  D @ x @ D.T   for every block.
    """
    if blocks.ndim != 3 or blocks.shape[1:] != (BLOCK, BLOCK):
        raise ValueError(f"dct2_batch expects (N, {BLOCK}, {BLOCK}), got {blocks.shape}")
    x = np.asarray(blocks, dtype=np.float64)
    # (n, i, l) = sum_{j, k} D[i, j] * x[n, j, k] * D.T[k, l]
    return np.einsum("ij,njk,kl->nil", D, x, D.T)


def idct2_batch(coeffs: np.ndarray) -> np.ndarray:
    """Inverse DCT applied to a stack of N blocks of shape (N, 8, 8).

    Computes  D.T @ X @ D   for every block.
    """
    if coeffs.ndim != 3 or coeffs.shape[1:] != (BLOCK, BLOCK):
        raise ValueError(f"idct2_batch expects (N, {BLOCK}, {BLOCK}), got {coeffs.shape}")
    X = np.asarray(coeffs, dtype=np.float64)
    # (n, i, l) = sum_{j, k} D.T[i, j] * X[n, j, k] * D[k, l]
    return np.einsum("ij,njk,kl->nil", D.T, X, D)
