# Tasks

- [x] Task 1: Set up project skeleton (directories, empty `__init__.py`, `requirements.txt`, README scaffold, asset folder).
  - [x] SubTask 1.1: Create the directory layout described in `spec.md` (`dct_core/`, `quality/`, `pipeline/`, `io_utils/`, `app/`, `tests/`, `benchmarks/`, `assets/`).
  - [x] SubTask 1.2: Write `requirements.txt` containing `numpy`, `Pillow`, `pytest` (pinned to known-good versions).
  - [x] SubTask 1.3: Add a placeholder `assets/sample.png` (or instructions to drop one in).
  - [x] SubTask 4: Run `pip install -r requirements.txt` to confirm the toolchain works on Windows.

- [x] Task 2: Implement the from-scratch 2-D DCT and inverse DCT in `dct_core/dct.py`.
  - [x] SubTask 2.1: Build the 8×8 DCT matrix `D` using the Type-II definition with the standard `alpha(k)` normalization.
  - [x] SubTask 2.2: Implement `dct2(block)` as `D @ x @ D.T` and `shape(batch) == (8, 8)`.
  - [x] SubTask 2.3: Implement `idct2(block)` as `D.T @ X @ D`.
  - [x] SubTask 2.4: Verify orthogonality numerically: `D @ D.T ≈ 2 * I`.

- [x] Task 3: Implement block splitting and merging in `dct_core/blocks.py`.
  - [x] SubTask 3.1: `split_blocks(img, block=8) -> (blocks, orig_shape, padded_shape)` with reflect padding when dimensions are not multiples of 8.
  - [x] SubTask 3.2: `merge_blocks(blocks, orig_shape, padded_shape) -> img` cropping padding back.

- [x] Task 4: Implement quantization with the standard JPEG tables in `dct_core/quant.py`.
  - [x] SubTask 4.1: Hard-code the standard JPEG luminance table at Q=50. (Chroma tables deferred to Task 7.)
  - [x] SubTask 4.2: Implement the Q-factor scaling formula per the JPEG standard and clamp `quant ∈ [1, 255]`.
  - [x] SubTask 4.3: Implement `quantize(coeffs, table)` and `dequantize(qcoeffs, table)` (round to nearest int on quantize).

- [x] Task 5: Implement quality metrics in `quality/metrics.py`.
  - [x] SubTask 5.1: `mse(a, b)` returning a scalar mean over all elements.
  - [x] SubTask 5.2: `psnr(a, b, max_val=255)` returning `10 * log10(max_val**2 / mse)` with an `inf` sentinel when MSE == 0.
  - [x] SubTask 5.3: `estimate_compressed_size(quantized_blocks, block=8)` using the run-length byte estimator described in `spec.md`.
  - [x] SubTask 5.4: `compression_ratio(orig_bytes, comp_bytes)`.

- [x] Task 6: **(50% Milestone)** Implement the grayscale CLI demo in `main.py`.
  - [x] SubTask 6.1: Accept a path argument (default `assets/sample.png`) and load it as grayscale via Pillow.
  - [x] SubTask 6.2: Apply level shift, split, DCT, quantize at Q=50, dequantize, inverse DCT, inverse level shift, clip, crop.
  - [x] SubTask 6.3: Save the reconstructed image as `recon_50.png` next to the input.
  - [x] SubTask 6.4: Print dimensions, original size, estimated compressed size, compression ratio, MSE, PSNR.
  - [x] SubTask 6.5: Verify the script runs end-to-end on `assets/sample.png` and produces reasonable numbers.

- [x] Task 7: Implement the full color pipeline in `pipeline/`.
  - [x] SubTask 7.1: `pipeline/color.py` — RGB ↔ YCbCr (BT.601) conversion functions.
  - [x] SubTask 7.2: `pipeline/pipeline.py` — `compress_image(img, quality)` orchestrating Y at Q and Cb/Cr at `Q // 2` (clamped ≥ 1).
  - [x] SubTask 7.3: `compress_image` returns a `CompressionResult` dataclass with `reconstructed`, `cr`, `mse_val`, `psnr_val`, `original_bytes`, `compressed_bytes`, `quality`, `chroma_quality`.

- [x] Task 8: Implement image I/O helpers in `io_utils/io.py`.
  - [x] SubTask 8.1: `load_image(path)` returning an `np.ndarray` and a flag indicating grayscale vs color.
  - [x] SubTask 8.2: `save_image(arr, path)` clipping and converting to uint8.
  - [x] SubTask 8.3: `make_comparison_grid(images, labels)` producing a horizontal montage with text labels via Pillow.

- [x] Task 9: Build the Tkinter GUI in `app/gui.py` (file named `gui.py` instead of `main.py` to avoid clashing with the CLI).
  - [x] SubTask 9.1: Layout: top toolbar (`Open Image...`), left preview (original), right preview (reconstructed), bottom metrics panel and slider.
  - [x] SubTask 9.2: Slider `1..100` default 75; debounce updates to keep latency under 1 s.
  - [x] SubTask 9.3: Show file size, compressed estimate, compression ratio, MSE, PSNR.
  - [x] SubTask 9.4: "Save 4-up Comparison Grid" button generates the 4-up grid via `make_comparison_grid` and saves it.

- [x] Task 10: Add the unit test suite under `tests/`.
  - [x] SubTask 10.1: `test_dct.py` — round-trip identity for random blocks and orthogonality check.
  - [x] SubTask 10.2: `test_quant.py` — Q=100 → quant table of all 1s; Q=1 → many entries ≥ 64.
  - [x] SubTask 10.3: `test_metrics.py` — known noise yields expected MSE and PSNR.
  - [x] SubTask 10.4: `test_pipeline.py` — grayscale E2E produces reasonable MSE/PSNR for a synthetic test image.
  - [x] SubTask 10.5: `test_color.py` and `test_io.py` cover the colour pipeline and PNG I/O round-trip.

- [x] Task 11: Add the benchmark sweep script `benchmarks/run_bench.py`.
  - [x] SubTask 11.1: Loop quality levels `[5, 25, 50, 75, 95]`, record CR, MSE, PSNR, compressed size.
  - [x] SubTask 11.2: Write `benchmarks/results.csv` and print a console table aligned with the syllabus "comparison table" example.

- [x] Task 12: Documentation and final polish.
  - [x] SubTask 12.1: `README.md` explaining theory (DCT, JPEG connection, MSE/PSNR formulas) and how to run the app, tests, and benchmarks.
  - [x] SubTask 12.2: A short "How to demo in class" subsection with the suggested verbal pitch from `spec.md`.
  - [x] SubTask 12.3: Sanity-verify the full app + benchmarks on `assets/sample.png` and capture a screenshot for the report.

# Task Dependencies
- [Task 2] depends on [Task 1].
- [Task 3] depends on [Task 2].
- [Task 4] depends on [Task 2].
- [Task 5] has no dependency (pure math).
- [Task 6] depends on [Task 2, Task 3, Task 4, Task 5]. **This is the 50% milestone.**
- [Task 7] depends on [Task 2, Task 3, Task 4, Task 5].
- [Task 8] depends on [Task 1].
- [Task 9] depends on [Task 7, Task 8].
- [Task 10] depends on [Task 2, Task 4, Task 5, Task 6].
- [Task 11] depends on [Task 7].
- [Task 12] depends on [Task 9, Task 10, Task 11].
