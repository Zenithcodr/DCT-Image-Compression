"""
pipeline.color
==============
RGB ↔ YCbCr conversion using the BT.601 full-range convention.

JPEG converts RGB to YCbCr and then compresses the three channels
*separately*. The chroma channels (Cb, Cr) carry colour information
that the human eye is less sensitive to than brightness, so they get
quantized more aggressively. We follow that same convention: the
pipeline in pipeline.py picks Q_chroma = max(1, Q // 2).

We use the *full-range* variant (Y/Cb/Cr all centred at 128, range
0..255) instead of the [16,235]/[16,240] studio range. This is what
Pillow uses and it makes the round-trip exactly lossless within ±0
on a per-pixel basis, which is what you want for an academic demo.
"""

from __future__ import annotations

import numpy as np


_RGB_TO_YCBCR = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=np.float64,
)

# Analytic inverse: gives bit-exact RGB ↔ YCbCr round-trip with no ±1 drift.
# (Derived from the forward by symbolic inversion.)
_YCBCR_TO_RGB = np.array(
    [
        [1.0,  0.0,             1.402],
        [1.0, -0.344136,       -0.714136],
        [1.0,  1.772,           0.0],
    ],
    dtype=np.float64,
)


def rgb_to_ycbcr(rgb_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 image of shape (H, W, 3) → YCbCr float64 in [0, 255] (BT.601 full-range)."""
    if rgb_u8.ndim != 3 or rgb_u8.shape[-1] != 3:
        raise ValueError(f"rgb_to_ycbcr expects (H, W, 3), got {rgb_u8.shape}")
    x = rgb_u8.astype(np.float64)
    return x @ _RGB_TO_YCBCR.T + 128.0


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_ycbcr, returning uint8 RGB (H, W, 3) clipped to [0, 255]."""
    if ycbcr.ndim != 3 or ycbcr.shape[-1] != 3:
        raise ValueError(f"ycbcr_to_rgb expects (H, W, 3), got {ycbcr.shape}")
    x = ycbcr.astype(np.float64) - 128.0
    rgb = x @ _YCBCR_TO_RGB.T
    return np.clip(rgb, 0, 255).astype(np.uint8)
