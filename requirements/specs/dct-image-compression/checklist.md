# Checklist

## 50% Milestone (must pass before considering the project half-done)

- [x] Project skeleton directories exist (`dct_core/`, `quality/`, `assets/`, `tests/`).
- [x] `requirements.txt` lists `numpy`, `Pillow`, `pytest`.
- [x] `dct_core/dct.py` exposes `dct2(block)` and `idct2(block)` using a from-scratch 8×8 DCT matrix.
- [x] `dct_core/blocks.py` exposes `split_blocks` and `merge_blocks` with reflect-padding for non-multiple-of-8 dimensions.
- [x] `dct_core/quant.py` embeds the standard JPEG luminance table and supports `quantize` / `dequantize` at any Q.
- [x] `quality/metrics.py` exposes `mse`, `psnr`, `compression_ratio`, and `estimate_compressed_size`.
- [x] `main.py` runs end-to-end on `assets/sample.png`:
  - [x] Loads the image as grayscale.
  - [x] Applies level shift, 8×8 DCT, quantization, dequantization, inverse DCT, inverse level shift.
  - [x] Saves a `*_recon_q<Q>_<mode>.png` next to the input.
  - [x] Prints image dimensions, original size, compressed size, CR, MSE, PSNR.
- [x] `tests/test_dct.py` round-trip test passes: `idct2(dct2(x)) ≈ x` within `1e-9`.
- [x] Demo produces visibly similar reconstruction at Q=50 on a natural image.

## 100% Milestone

- [x] `pipeline/pipeline.py` implements the full RGB pipeline: RGB → YCbCr → per-channel DCT/quant/IDCT → YCbCr → RGB.
- [x] Cb and Cr use `Q_chroma = max(1, Q // 2)`.
- [x] `io_utils/io.py` exposes `load_image`, `save_image`, `make_comparison_grid` with labelled images.
- [x] Tkinter GUI launches via `python app/gui.py` and shows original + reconstructed previews.
- [x] Quality slider 1..100 updates reconstructed preview and metrics within ~1 s on a 256×256 image.
- [x] Metrics panel shows original size, compressed size estimate, compression ratio, MSE, PSNR.
- [x] "Save Reconstruction" button writes a PNG to disk.
- [x] "Save 4-up Comparison Grid" button writes `Original | Q=90 | Q=50 | Q=10` to disk.
- [x] `benchmarks/run_bench.py` runs a quality sweep and writes `benchmarks/results.csv` with columns `quality, channel, mse, psnr_db, compression_ratio, original_kb, compressed_kb`.
- [x] All tests in `tests/` pass under `pytest` (24/24).

## Quality & Theory Correctness

- [x] DCT matrix `D` satisfies `D @ D.T ≈ I` (orthogonal under our normalization; the docstring of `dct_core/dct.py` notes the equivalent 2·I form under the standard normalization).
- [x] At Q=100, every quantizer entry equals 1; reconstructed image differs from original by at most 1 in any pixel.
- [x] At Q=1 on a natural image, MSE is noticeably higher and visible 8×8 artifacts are obvious in the reconstruction.
- [x] `psnr(a, b)` returns `inf` when `mse == 0`.
- [x] `psnr(a, b)` for Gaussian noise with sigma=10 matches `≈ 28 dB` within ±2 dB.
- [x] `compression_ratio` is always ≥ 0; positive for any meaningful compressed output.

## Documentation & Demo

- [x] `README.md` exists, explains the JPEG connection, and shows the verbal pitch from `spec.md`.
- [x] A reproducible command (`python main.py assets/sample.png` and `python app/gui.py`) is documented.
- [x] A demo script for the teacher (load → slide 90 → slide 50 → slide 10 → show metrics → show comparison grid) is included in the README.
- [x] No external network calls, no copyrighted sample images bundled; sample images are synthetically generated and clearly named (e.g. `assets/sample_color.png`).
