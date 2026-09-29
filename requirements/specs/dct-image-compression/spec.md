# DCT-Based Image Compression & Quality Analysis — Spec

## Why
A Digital Image Processing (DIP) project must go beyond "compressing an image." It must expose the actual signal-processing pipeline — block splitting, forward Discrete Cosine Transform (DCT), quantization, inverse DCT, reconstruction — and then quantify the trade-off between storage size and visual fidelity using Compression Ratio, Mean Squared Error (MSE), and Peak Signal-to-Noise Ratio (PSNR). This project delivers an interactive experimental system that lets the user move a quality slider and immediately observe how compression level affects file size, MSE, PSNR, and visible artifacts.

## What Changes
- New Python application `dct_compress/` implementing the full DCT compression pipeline (grayscale + YCbCr color).
- New module `dct_core/` with pure NumPy implementation of 2-D DCT, 8×8 block processing, standard JPEG luminance/chrominance quantization matrices, quantization, dequantization, and inverse DCT.
- New module `quality/` computing Compression Ratio, MSE, PSNR, and per-block visual accuracy.
- New module `pipeline/` orchestrating: load → color conversion → level shift → block split → DCT → quantize → (RLE) → dequantize → inverse DCT → inverse level shift → reconstruct.
- New module `io_utils/` handling image loading (PNG/JPG/BMP), saving reconstructed image, and exporting comparison grids.
- New Tkinter desktop UI `app/` providing: file picker, original preview, quality slider (1–100, mapped to JPEG Q-factor), live reconstruction preview, side-by-side comparison grid (Original | Q=high | Q=medium | Q=low), numeric metrics panel, and "save reconstructed" button.
- New unit-test suite under `tests/` verifying mathematical correctness of DCT/IDCT, quantization/dequantization round-trip, metric formulas, and end-to-end pipeline.
- New sample image fixtures and reproducible benchmark script `benchmarks/run_bench.py` producing a level-sweep CSV/table.
- New documentation `README.md` describing theory, JPEG connection, and how to run the demo.

## Impact
- Affected specs: image compression models (Unit 4), transforms (Unit 1), image quality metrics.
- Affected code: entirely new project. No existing files in the repository to modify.

## ADDED Requirements

### Requirement: Project Structure & Tooling
The system SHALL provide a clean Python project with:
- Python 3.10+.
- Dependencies: `numpy`, `Pillow`, `tkinter` (stdlib on most platforms).
- Dev dependencies: `pytest`.
- Layout: `dct_core/`, `quality/`, `pipeline/`, `io_utils/`, `app/`, `tests/`, `benchmarks/`, `assets/`.
- The project SHALL run from source with `python -m app.main` (or equivalent entry point) and tests SHALL run with `pytest`.

#### Scenario: Fresh clone boots
- **WHEN** a user clones the repo, installs deps, and runs `python -m app.main`
- **THEN** the GUI launches and shows the empty state with a "Load image" button.

#### Scenario: Tests run
- **WHEN** a user runs `pytest`
- **THEN** all unit tests for DCT/IDCT, metrics, and pipeline pass.

### Requirement: 2-D DCT & IDCT (Mathematical Correctness)
The system SHALL provide a from-scratch 2-D Type-II DCT and the matching 2-D Type-III inverse DCT implemented in NumPy (no `scipy.fft` shortcuts).
- The forward DCT SHALL accept an `8×8` float block and return an `8×8` coefficient block per the standard separable DCT-II definition.
- The inverse DCT SHALL accept the coefficient block and recover the original block within floating-point tolerance (`max abs error < 1e-6`).
- An `orthogonality` invariant SHALL hold: `DCT(IDCT(x)) == x` for arbitrary blocks.

#### Scenario: Round-trip identity
- **WHEN** any 8×8 random block is passed through `forward_dct` and then `inverse_dct`
- **THEN** the output equals the input up to tolerance `1e-6`.

### Requirement: Block Processing
The system SHALL split an input 2-D image (or a single color channel) into non-overlapping `8×8` blocks, apply DCT per block, and reassemble reconstructed blocks into an output image of the same shape.
- Edge cases: images whose dimensions are not multiples of 8 SHALL be padded (reflect or zero-pad) before processing and cropped back to original dimensions after reconstruction.

#### Scenario: Non-multiple-of-8 dimensions
- **WHEN** a 515×513 grayscale image is processed
- **THEN** the reconstructed image has dimensions 515×513 and matches the padded-then-cropped inverse pipeline.

### Requirement: Level Shift
The system SHALL subtract 128 from pixel intensities (level shift to signed range `[-128, 127]`) before DCT and add 128 back after inverse DCT, as JPEG does.

#### Scenario: Pixel value range preserved
- **WHEN** any uint8 image is compressed and reconstructed with Q-factor 100 (lossless quantization table)
- **THEN** reconstructed pixel values lie in `[0, 255]` (clipped if necessary) and differ from the original only by quantization rounding.

### Requirement: Standard Quantization Matrices
The system SHALL embed the standard JPEG luminance (`Qt_lum`) and chrominance (`Qt_chr`) quantization tables at quality factor 50, and SHALL scale both tables by the standard JPEG `quality` scaling rule for arbitrary Q-factor 1..100:
- `scale = 5000 / Q` if `Q < 50`, else `200 - 2*Q` if `Q ≥ 50`, clamped to `[1, 255]`.
- `quant = floor((base * scale + 50) / 100)`, clamped to `[1, 255]`.

#### Scenario: Q=100 round-trip
- **WHEN** Q-factor 100 is selected
- **THEN** every quantizer entry equals 1 and the reconstructed image differs from the original by at most 1 in any pixel (pure rounding).

#### Scenario: Q=1 high compression
- **WHEN** Q-factor 1 is selected
- **THEN** many quantizer entries are large (≥ 64), the reconstruction shows visible 8×8 blocking, and PSNR is typically below 25 dB on natural images.

### Requirement: Quantization & Dequantization
The system SHALL provide `quantize(coeffs, table)` and `dequantize(qcoeffs, table)` operations that perform element-wise division / multiplication by the table, rounding to nearest integer on quantize.

#### Scenario: Quantization idempotency
- **WHEN** coefficients are quantized then dequantized with the same table
- **THEN** the resulting values differ from the original coefficients by at most `table / 2` per element.

### Requirement: Color Pipeline (YCbCr)
The system SHALL support color images by:
1. Converting RGB → YCbCr (`ITU-R BT.601`).
2. Compressing Y at the user-selected quality Q.
3. Compressing Cb and Cr at a reduced chrominance quality `Q_chroma = max(1, Q // 2)` (standard JPEG subsampling simplification; full 4:2:0 downsampling is out of scope at v1 but Cb/Cr are still quantized more aggressively to match JPEG practice).
4. Reconstructing each channel separately, then merging YCbCr → RGB.

#### Scenario: Color image compressed and reconstructed
- **WHEN** a color RGB image is loaded and any quality is selected
- **THEN** a reconstructed RGB image of identical size is returned.

### Requirement: Quality Metrics
The system SHALL compute:
- **Compression Ratio (CR)** = `original_bytes / compressed_bytes`, where `compressed_bytes` is the estimated number of bytes if every non-zero quantized value is stored as 1 sign byte + 1 magnitude byte and zero-runs are run-length coded (length byte + sign byte). A simplified estimator SHALL be exposed (`nc0`) and documented.
- **MSE** = `mean((orig - recon) ** 2)` over all pixels (and per channel when requested).
- **PSNR** = `10 * log10(MAX^2 / MSE)` where `MAX = 255` for uint8 images. If `MSE == 0`, return `inf` (or `99.99` for display).

#### Scenario: Identical images
- **WHEN** original and reconstructed images are identical
- **THEN** MSE == 0 and PSNR is reported as `inf`.

#### Scenario: Random noise vs known distortion
- **WHEN** a known Gaussian noise (sigma=10) is added to an image
- **THEN** MSE ≈ 100 and PSNR ≈ 28 dB (within ±2 dB), matching the closed-form expectation.

### Requirement: GUI Application
The system SHALL provide a Tkinter desktop app with:
- File picker (`Load image` button) accepting `.png`, `.jpg`, `.jpeg`, `.bmp`.
- Original image preview (scaled to fit a 512×512 viewport).
- Quality slider `1..100` (default 75) labeled "Compression Level".
- Reconstructed image preview that updates within ~1 s for a 1024×1024 image on a typical laptop.
- Metrics panel showing file size, compressed size estimate, compression ratio, MSE, and PSNR for the current quality.
- "Save reconstructed" button writing a PNG to disk.
- "Compare Levels" button generating a side-by-side grid `Original | Q=90 | Q=50 | Q=10` and saving it as `comparison_<name>.png`.
- Status bar showing progress during compression.

#### Scenario: Load and compress
- **WHEN** the user loads a 1024×1024 color image and moves the slider to 50
- **THEN** the reconstructed preview, metrics, and file-size estimate update within 1 second.

#### Scenario: Generate comparison grid
- **WHEN** the user clicks "Compare Levels"
- **THEN** a 4-up grid PNG is written to disk and a confirmation dialog shows the file path.

### Requirement: 50% Milestone (Demo-Ready Core)
The 50% milestone SHALL be a fully runnable demo of `Image → 8×8 blocks → DCT → quantization → inverse DCT → reconstructed image` for **grayscale** images, with hard-coded Q=50 and printed MSE / PSNR / Compression Ratio. No GUI is required at the 50% milestone; a CLI script suffices. See "50% Milestone Specification" section at the end of this document.

### Requirement: 100% Milestone (Full Delivery)
The 100% milestone SHALL include everything above: GUI, color pipeline, variable quality slider, comparison grid, and full test suite.

## MODIFIED Requirements
None — this is a greenfield project.

## REMOVED Requirements
None.

---

## Architecture & Technical Choices

### Language & Libraries
- **Python 3.10+** — universally available, easy to demo on a laptop.
- **NumPy** — vectorized block operations.
- **Pillow (PIL)** — image I/O and color conversions.
- **Tkinter** — stdlib GUI, no extra install on Windows / macOS; works on Linux.
- **pytest** — unit testing.
- **No OpenCV, no scipy.fft** — to keep the math visible and aligned with syllabus.

### DCT Math (Type-II)
For an 8×8 block `x[i, j]`:

```
X(u, v) = (1/4) * c(u) * c(v) * sum_{i,j=0..7} x(i,j) *
          cos((2i+1)u*pi/16) * cos((2j+1)v*pi/16)
```

with `c(k) = 1/sqrt(2)` when `k == 0`, else `1`.

Implementation: a precomputed `D` matrix of shape `(8, 8)` whose entry `D[k, i] = alpha(k) * cos((2i+1)k*pi/16)`. Then `X = D @ x @ D.T`. Inverse: `x = D.T @ X @ D`.

### JPEG Standard Quantization Tables (Q=50)

```
16  11  10  16  24  40  51  61
12  12  14  19  26  58  60  55
14  13  16  24  40  57  69  56
14  17  22  29  51  87  80  62
18  22  37  56  68 103  77  62
30  35  55  64  81 104  92  75
50  72  78  87 103 121 100  78
68  92  95  98 112 100 103  99   (luminance)

17  18  24  47  99 100 117 109
18  21  26  66  99 100 117 109
24  26  56  99 100 117 109 109
47  66  99 100 117 109 109 109
99  99 100 117 109 109 109 109
99 100 117 109 109 109 109 109
99 117 109 109 109 109 109 109
99 109 109 109 109 109 109 109   (chrominance)
```

### Compression Ratio Estimator
Estimate `compressed_bytes` as the number of bytes needed to serialize:
- For each 8×8 block in raster order: count non-zero quantized coefficients after a zig-zag scan and store `(run_length, value)` pairs.
- Each pair uses 1 byte for run-length (cap at 255) and 1 byte for the value (assume values in `[-127, 127]` mapped to 0–254).

This is intentionally a teaching estimator, not a Huffman coder. It still responds to compression level (more zeros at high compression → smaller file).

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `dct_core/dct.py` | `dct2(block)`, `idct2(block)`, 8×8 DCT matrix `D`. |
| `dct_core/quant.py` | JPEG Q-factor scaling, quantize/dequantize, standard tables. |
| `dct_core/blocks.py` | `split_blocks(img)`, `merge_blocks(blocks, shape)`. |
| `pipeline/pipeline.py` | `compress_image(image, quality) -> {reconstructed, cr, mse, psnr, compressed_bytes}`. |
| `pipeline/color.py` | RGB ↔ YCbCr, per-channel orchestration. |
| `quality/metrics.py` | `mse(a, b)`, `psnr(a, b)`, `compression_ratio(orig_bytes, comp_bytes)`, `estimate_compressed_size(quantized)`. |
| `io_utils/io.py` | `load_image(path)`, `save_image(arr, path)`, `make_comparison_grid(origs, recons)`. |
| `app/main.py` | Tkinter UI wiring slider → pipeline → preview/metrics. |
| `tests/` | Unit tests for DCT, IDCT, quant, metrics, pipeline. |
| `benchmarks/run_bench.py` | Sweep quality 10, 25, 50, 75, 90 → CSV/console table of results. |

### Test Strategy
- **Math correctness**: random 8×8 block → DCT → IDCT → equality.
- **Orthogonality**: `D @ D.T ≈ 2 * I` (constant factor).
- **Quantization round-trip**: per-element error ≤ table/2.
- **Metric sanity**: known noise → expected MSE/PSNR.
- **Pipeline E2E**: small known image → expected approximate metrics.

---

## 50% Milestone Specification

The 50% milestone is the **tomorrow-defensible** deliverable: a runnable command-line demo proving the DCT compression pipeline works end-to-end on grayscale images, with hard-coded quality and printed metrics. No GUI, no color, no slider, no comparison grid — just the signal-processing pipeline that the syllabus requires.

### 50% Functional Requirements
1. **Load** a grayscale image from a path passed on the command line (default `assets/sample.png`).
2. **Convert** to float32, apply level shift (`−128`).
3. **Split** into 8×8 blocks, pad if needed.
4. **Forward DCT** each block using the from-scratch 2-D DCT.
5. **Quantize** with the standard JPEG luminance table at fixed quality Q = 50.
6. **Dequantize** and apply **inverse DCT**.
7. **Inverse level shift** and clip to `[0, 255]`, crop padding.
8. **Save** the reconstructed image as `recon_50.png` next to the input.
9. **Print** to stdout: image dimensions, original file size, estimated compressed size, compression ratio, MSE, PSNR.

### 50% Acceptance
- Running `python main.py assets/sample.png` produces:
  - A file `recon_50.png` visually similar to the original.
  - Printed output similar to:
    ```
    Image:           512 x 512
    Original size:   234.5 KB
    Compressed est.: 71.2 KB
    Compression ratio: 3.29
    MSE:             6.83
    PSNR:            39.79 dB
    ```

### 50% Project Layout (Minimal)
```
DIP project/
├── main.py                  # CLI demo entry point
├── dct_core/
│   ├── __init__.py
│   ├── dct.py               # dct2, idct2
│   ├── quant.py             # JPEG luminance table + quantize/dequantize
│   └── blocks.py            # split/merge
├── quality/
│   ├── __init__.py
│   └── metrics.py           # mse, psnr, compression_ratio
├── assets/
│   └── sample.png           # any 512×512 grayscale test image
└── tests/
    └── test_dct.py          # round-trip test for the 50% demo
```

### 50% — What This Demonstrates to the Teacher Tomorrow
- The **entire DIP signal-processing chain** is implemented from scratch: block split → DCT → quantize → dequantize → IDCT → reconstruction.
- **JPEG is explained as a real-world example**: "we use the same standard luminance quantization table that JPEG uses."
- **Numerical quality is measured** with MSE and PSNR — not claimed, computed.
- **Compression ratio is computed**, not guessed.
- The reconstructed image visibly resembles the original at Q=50.

This alone is a complete, defensible 50% of the project. The remaining 50% (color, GUI, slider, comparison grid, tests) is straightforward extension work that does not change the underlying pipeline already demoed.