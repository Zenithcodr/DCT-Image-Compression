"""
io_utils.io
===========
Image I/O helpers and a small "compare-levels" grid builder.

load_image:   PNG/JPG/BMP → numpy uint8 array, auto-detects colour vs grayscale.
save_image:   numpy uint8 → PNG on disk.
make_comparison_grid: 4-up labelled montage, e.g. Original | Q=90 | Q=50 | Q=10.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_image(path: str) -> Tuple[np.ndarray, bool]:
    """Return (image_uint8, is_color). Color images come back as (H, W, 3) RGB."""
    img = Image.open(path)
    if img.mode == "L":
        return np.asarray(img, dtype=np.uint8), False
    rgb = img.convert("RGB")
    return np.asarray(rgb, dtype=np.uint8), True


def save_image(arr: np.ndarray, path: str) -> str:
    """Save a uint8 (H, W) or (H, W, 3) array as PNG or JPG. Returns absolute path."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        img = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"save_image expects (H, W) or (H, W, 3), got {arr.shape}")
    out_path = os.path.abspath(path)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        img.save(out_path, quality=100)
    else:
        img.save(out_path)
    return out_path


def _to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr, mode="RGB")


def _get_font(size: int):
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_comparison_grid(images: Iterable[np.ndarray], labels: Iterable[str],
                         out_path: str, tile_size: int = 256,
                         bg_color: Tuple[int, int, int] = (255, 255, 255),
                         label_color: Tuple[int, int, int] = (20, 20, 20)) -> str:
    """Build a horizontal montage of N images with text labels above each tile.

    All input images are resized to (tile_size, tile_size) for visual parity.
    Returns the absolute output path.
    """
    images = list(images)
    labels = list(labels)
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if not images:
        raise ValueError("at least one image is required")

    pad = 12
    label_h = 36
    W = len(images) * tile_size + (len(images) + 1) * pad
    H = tile_size + 2 * pad + label_h
    canvas = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(canvas)
    font = _get_font(18)

    for i, (img, label) in enumerate(zip(images, labels)):
        pil = _to_pil(img).resize((tile_size, tile_size), Image.LANCZOS)
        x = pad + i * (tile_size + pad)
        canvas.paste(pil, (x, pad))
        # Centre the label above the tile.
        try:
            tw = draw.textlength(label, font=font)
        except AttributeError:
            tw = len(label) * 9
        tx = x + (tile_size - tw) // 2
        draw.text((tx, pad + tile_size + 6), label, fill=label_color, font=font)

    return save_image(np.asarray(canvas), out_path)
