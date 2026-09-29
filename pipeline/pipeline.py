"""
pipeline.pipeline
=================
Orchestrates the full DCT compression pipeline for both grayscale and
colour images.

For colour images we go RGB → YCbCr → per-channel DCT/quant/IDCT
→ YCbCr → RGB. The luminance channel Y is processed at the user-selected
quality Q; the chroma channels Cb and Cr are processed at
Q_chroma = max(1, Q // 2), matching how JPEG treats chroma more
aggressively than luma.

The pipeline is vectorized: every channel is processed as one batch
of (N, 8, 8) blocks so the demo stays fast on a typical laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dct_core.blocks import merge_blocks, split_blocks
from dct_core.dct import dct2_batch, idct2_batch
from dct_core.quant import (
    LUM_TABLE_Q50,
    scale_table,
    quantize,
    dequantize,
)
from quality.metrics import (
    compression_ratio,
    estimate_compressed_size,
    mse,
    psnr,
)


@dataclass
class CompressionResult:
    reconstructed: np.ndarray            # uint8 image with the same shape as input
    cr: float                            # compression ratio (orig / compressed)
    mse_val: float                       # mean squared error over all pixels
    psnr_val: float                      # decibels, +inf if lossless
    original_bytes: int                  # raw pixel bytes (W * H * C)
    compressed_bytes: int                # zig-zag / RLE estimator result
    quality: int                         # Q used for luminance
    chroma_quality: int                  # Q used for chroma channels


def _compress_channel(channel: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress one 2-D channel (H, W) at quality Q and return (recon, qblocks, table)."""
    shifted = channel.astype(np.float64) - 128.0
    blocks, orig_shape, padded_shape = split_blocks(shifted)
    coeffs = dct2_batch(blocks)
    table = scale_table(LUM_TABLE_Q50, q)
    qblocks = quantize(coeffs, table)
    rec_coeffs = dequantize(qblocks, table)
    rec_shifted = idct2_batch(rec_coeffs)
    rec = merge_blocks(rec_shifted, orig_shape, padded_shape) + 128.0
    return rec, qblocks, table


def compress_image(image_u8: np.ndarray, quality: int = 75) -> CompressionResult:
    """Top-level entry: accept (H, W) grayscale or (H, W, 3) RGB uint8."""
    if not (1 <= quality <= 100):
        raise ValueError(f"quality must be in [1, 100], got {quality}")
    if image_u8.dtype != np.uint8:
        image_u8 = image_u8.astype(np.uint8)

    if image_u8.ndim == 2:
        rec_f, qblocks, _ = _compress_channel(image_u8, quality)
        rec_u8 = np.clip(rec_f, 0, 255).astype(np.uint8)
        original_bytes = int(image_u8.size)
        compressed_bytes = estimate_compressed_size(qblocks)
        cr = compression_ratio(original_bytes, compressed_bytes)
        m = mse(image_u8.astype(np.float64), rec_u8.astype(np.float64))
        p = psnr(image_u8.astype(np.float64), rec_u8.astype(np.float64))
        return CompressionResult(rec_u8, cr, m, p, original_bytes, compressed_bytes,
                                 quality, quality)

    if image_u8.ndim == 3 and image_u8.shape[-1] == 3:
        from .color import rgb_to_ycbcr, ycbcr_to_rgb
        ycbcr = rgb_to_ycbcr(image_u8)
        y_q = quality
        c_q = max(1, quality // 2)
        y_rec, y_qblocks, _ = _compress_channel(ycbcr[..., 0], y_q)
        cb_rec, cb_qblocks, _ = _compress_channel(ycbcr[..., 1], c_q)
        cr_rec, cr_qblocks, _ = _compress_channel(ycbcr[..., 2], c_q)
        rec_ycbcr = np.stack([y_rec, cb_rec, cr_rec], axis=-1)
        rec_u8 = ycbcr_to_rgb(rec_ycbcr)
        original_bytes = int(image_u8.size)
        compressed_bytes = (
            estimate_compressed_size(y_qblocks)
            + estimate_compressed_size(cb_qblocks)
            + estimate_compressed_size(cr_qblocks)
        )
        cr = compression_ratio(original_bytes, compressed_bytes)
        m = mse(image_u8.astype(np.float64), rec_u8.astype(np.float64))
        p = psnr(image_u8.astype(np.float64), rec_u8.astype(np.float64))
        return CompressionResult(rec_u8, cr, m, p, original_bytes, compressed_bytes,
                                 y_q, c_q)

    raise ValueError(f"compress_image expects (H, W) or (H, W, 3), got {image_u8.shape}")
