"""
benchmarks.run_bench
====================
Sweep across a range of JPEG quality factors and dump a CSV with
(Q, channel, mse, psnr, cr, orig_kb, comp_kb) rows.

Usage:
    python -m benchmarks.run_bench path/to/image.png
    python -m benchmarks.run_bench path/to/image.png --qs 5 10 25 50 75 90

If no path is given and assets/sample.png exists, that is used.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

import numpy as np

from io_utils.io import load_image
from pipeline import compress_image


def _fmt_psnr(p: float) -> str:
    return "inf" if p == float("inf") else f"{p:.4f}"


def run(image_path: str, qs: List[int], out_csv: str) -> int:
    img, is_color = load_image(image_path)
    print(f"Benchmarking {image_path}  ({'colour' if is_color else 'grayscale'}, "
          f"shape={img.shape})")

    rows = []
    for q in qs:
        r = compress_image(img, quality=q)
        channel = "rgb" if is_color else "gray"
        rows.append({
            "quality": q,
            "channel": channel,
            "mse": f"{r.mse_val:.4f}",
            "psnr_db": _fmt_psnr(r.psnr_val),
            "compression_ratio": f"{r.cr:.4f}",
            "original_kb": f"{r.original_bytes / 1024:.2f}",
            "compressed_kb": f"{r.compressed_bytes / 1024:.2f}",
        })
        print(f"  Q={q:3d}  CR={r.cr:5.2f}x  "
              f"MSE={r.mse_val:8.3f}  PSNR={_fmt_psnr(r.psnr_val):>10s}")

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {os.path.abspath(out_csv)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark DCT compression over Q values.")
    parser.add_argument("path", nargs="?",
                        default=os.path.join("assets", "sample.png"))
    parser.add_argument("--qs", nargs="*", type=int,
                        default=[10, 25, 50, 75, 90])
    parser.add_argument("--out", default=os.path.join("benchmarks", "results.csv"))
    args = parser.parse_args(argv)
    if not os.path.isfile(args.path):
        print(f"No image at {args.path}. Place one there or pass a path.")
        return 0
    return run(args.path, args.qs, args.out)


if __name__ == "__main__":
    sys.exit(main())
