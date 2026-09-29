"""
dct_core.blocks
===============
Helpers for splitting a 2-D image into non-overlapping 8x8 blocks
(the way JPEG does it) and stitching them back together.

If the image height or width is not a multiple of 8, we reflect-pad
the edges so the DCT can run on clean 8x8 tiles, then crop the padding
back when we reconstruct. This keeps the reconstruction exactly the
same size as the input.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .dct import BLOCK


def _padded_shape(h: int, w: int, block: int = BLOCK) -> Tuple[int, int]:
    return ((h + block - 1) // block) * block, ((w + block - 1) // block) * block


def split_blocks(img: np.ndarray, block: int = BLOCK):
    """Split a (H, W) image into (N, block, block) blocks.

    Returns (blocks, orig_shape, padded_shape).
    Padding (reflect mode) is applied if H or W is not a multiple of `block`.
    """
    if img.ndim != 2:
        raise ValueError(f"split_blocks expects a 2-D image, got shape {img.shape}")
    h, w = img.shape
    hp, wp = _padded_shape(h, w, block)
    if (hp, wp) != (h, w):
        pad_h = hp - h
        pad_w = wp - w
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
    else:
        padded = img
    n_rows = hp // block
    n_cols = wp // block
    blocks = padded.reshape(n_rows, n_cols, block, block)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(n_rows * n_cols, block, block)
    return blocks, (h, w), (hp, wp)


def merge_blocks(blocks: np.ndarray, orig_shape, padded_shape, block: int = BLOCK) -> np.ndarray:
    """Inverse of split_blocks: stitch (N, block, block) blocks back into an image
    and crop padding back to orig_shape."""
    if blocks.ndim != 3 or blocks.shape[1:] != (block, block):
        raise ValueError(f"merge_blocks expects (N, {block}, {block}), got {blocks.shape}")
    hp, wp = padded_shape
    n_rows = hp // block
    n_cols = wp // block
    if blocks.shape[0] != n_rows * n_cols:
        raise ValueError(
            f"merge_blocks: {blocks.shape[0]} blocks do not fit padded shape {padded_shape}"
        )
    grid = blocks.reshape(n_rows, block, n_cols, block).transpose(0, 2, 1, 3).reshape(hp, wp)
    h, w = orig_shape
    return grid[:h, :w]
