"""
DCT-Based Image Compression — full pipeline demo.

Run from the project root:

    python main.py                         # uses assets/sample.png, grayscale
    python main.py my_photo.jpg            # any image path
    python main.py path/to/photo.png --q 75
    python main.py my_photo.jpg --color    # use YCbCr pipeline (default if input is RGB)
    python main.py my_photo.jpg --grid     # also save Original | Q=90 | Q=50 | Q=10 grid

What this script does (the complete DIP signal-processing chain):

    Image  →  grayscale or RGB → YCbCr (if colour)
        →  level shift  →  split 8x8 blocks
        →  2-D DCT  →  quantize (JPEG luminance table @ Q)
        →  dequantize  →  inverse DCT  →  inverse level shift
        →  reconstructed image

It then prints MSE, PSNR, and Compression Ratio so the trade-off
between file size and image quality is *measured*, not guessed.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
from PIL import Image

from io_utils.io import load_image, make_comparison_grid, save_image
from pipeline import compress_image


def _print_friendly_missing(path: str) -> None:
    print(f"No image found at '{path}'.")
    print("Tip: run  python main.py path/to/your_photo.jpg")
    print("     (any PNG / JPG / BMP; colour images are auto-converted)")


def run(path: str, q: int, color: bool, grid: bool, fmt: str) -> int:
    if not os.path.isfile(path):
        _print_friendly_missing(path)
        return 0

    img, is_color = load_image(path)
    if not color and img.ndim == 3:
        # User asked for grayscale explicitly.
        img = np.asarray(Image.fromarray(img, mode="RGB").convert("L"))
        is_color = False

    result = compress_image(img, quality=q)
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    base, _ = os.path.splitext(os.path.basename(path))
    mode_tag = "color" if is_color else "gray"
    out_path = os.path.join(out_dir, f"{base}_recon_q{q}_{mode_tag}.{fmt}")
    save_image(result.reconstructed, out_path)

    psnr_str = "inf" if result.psnr_val == float("inf") else f"{result.psnr_val:.2f} dB"
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    print("-" * 50)
    print("  DCT Image Compression")
    print("-" * 50)
    print(f"  Image:            {w} x {h}  "
          f"({'colour' if is_color else 'grayscale'})")
    print(f"  Quality factor Q: {q}")
    print(f"  Luma Q:           {result.quality}")
    print(f"  Chroma Q:         {result.chroma_quality}")
    print(f"  Original size:    {result.original_bytes / 1024:.2f} KB")
    print(f"  Compressed est.:  {result.compressed_bytes / 1024:.2f} KB")
    print(f"  Compression ratio:{result.cr:7.2f} x")
    print(f"  MSE:              {result.mse_val:7.3f}")
    print(f"  PSNR:             {psnr_str}")
    print("-" * 50)
    print(f"  Saved reconstructed image to: {out_path}")

    if grid:
        levels = [90, 50, 10]
        images = [img]
        labels = ["Original"]
        for qq in levels:
            r = compress_image(img, quality=qq)
            images.append(r.reconstructed)
            labels.append(f"Q = {qq}")
        grid_path = os.path.join(out_dir, f"{base}_comparison.{fmt}")
        make_comparison_grid(images, labels, grid_path, tile_size=256)
        print(f"  Saved 4-up comparison grid to: {grid_path}")

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="DCT-based image compression demo (grayscale or colour)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=os.path.join("assets", "sample.png"),
        help="Path to an input image (PNG / JPG / BMP). Defaults to assets/sample.png.",
    )
    parser.add_argument(
        "-q", "--quality",
        dest="quality",
        type=int,
        default=75,
        help="JPEG quality factor Q in [1, 100] (default: 75).",
    )
    parser.add_argument(
        "--color", dest="color", action="store_true",
        help="Use the colour (RGB → YCbCr) pipeline. Default if the input image is RGB.",
    )
    parser.add_argument(
        "--no-color", dest="color", action="store_false",
        help="Force grayscale pipeline even if the input is RGB.",
    )
    parser.add_argument(
        "--grid", dest="grid", action="store_true",
        help="Also save a 4-up comparison grid (Original | Q=90 | Q=50 | Q=10).",
    )
    parser.add_argument(
        "--format", dest="format", type=str, choices=["png", "jpg"], default="png",
        help="Output image format (png or jpg). Default is png.",
    )
    parser.set_defaults(color=None)
    args = parser.parse_args(argv)
    color_flag = args.color if args.color is not None else True
    return run(args.path, args.quality, color_flag, args.grid, args.format)


if __name__ == "__main__":
    sys.exit(main())
