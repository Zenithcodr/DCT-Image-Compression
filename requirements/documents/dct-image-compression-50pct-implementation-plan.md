# Plan — Implement 50% Milestone of DCT Image Compression Project

## Summary
Implement Tasks 1 → 6 from `.trae/specs/dct-image-compression/tasks.md`, which together form the runnable 50% deliverable specified in `.trae/specs/dct-image-compression/spec.md` (the "50% Milestone Specification" section). After each task, append a simple-words explanation to a new `PROGRESS.md` so the user always knows what was built, how it works, and what is left.

Out of scope for this plan: Tasks 7–12 (color pipeline, GUI, full test suite, benchmarks, polished README). Those belong to the 100% milestone and are explicitly **not** being done in this round.

## Current State Analysis
- `.trae/specs/dct-image-compression/spec.md` — full spec, written.
- `.trae/specs/dct-image-compression/tasks.md` — full task list, written.
- `.trae/specs/dct-image-compression/checklist.md` — verification list, written.
- The 8 source directories (`dct_core/`, `quality/`, `pipeline/`, `io_utils/`, `app/`, `tests/`, `benchmarks/`, `assets/`) exist but are empty.
- No Python files yet; no `requirements.txt`; no `assets/sample.png` (user-supplied).
- Python on the host machine: TBD at first run; `python --version` will be checked.

## Proposed Changes

### Task 1 — Project Skeleton
- Create `requirements.txt` with pinned versions: `numpy>=1.24,<2.1`, `Pillow>=10.0,<11.0`, `pytest>=7.4`.
- Create empty `__init__.py` in `dct_core/` and `quality/` (the only two Python packages at 50%).
- Create `assets/.gitkeep` so the directory survives in version control. **Do not** add any sample image (user will supply it; CLI must be friendly about missing file).
- Verify `python -m pip install -r requirements.txt` succeeds on Windows.

### Task 2 — `dct_core/dct.py`
- Build the 8×8 DCT matrix `D` from the Type-II formula:
  `D[k, i] = alpha(k) * cos((2i + 1) * k * π / 16)` where `alpha(0) = 1/√8`, `alpha(k>0) = √(2/8) = 1/2`.
- Implement `dct2(block: np.ndarray) -> np.ndarray` as `D @ block @ D.T`. Input/output shapes `(8, 8)`. Use `np.float64`.
- Implement `idct2(coeffs: np.ndarray) -> np.ndarray` as `D.T @ coeffs @ D`.
- Self-document with a short module docstring explaining the math in plain English.
- Why this design: it matches the math in `spec.md` exactly; separable matrix product is what the spec calls out; uses no `scipy` shortcut so the transform is visible.

### Task 3 — `dct_core/blocks.py`
- `split_blocks(img: np.ndarray, block: int = 8) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]`
  - Pad with `np.pad(..., mode="reflect")` if H or W is not a multiple of 8.
  - Reshape into `(n_blocks, 8, 8)`.
  - Return `(blocks, (H, W), (Hp, Wp))`.
- `merge_blocks(blocks: np.ndarray, orig_shape: tuple, padded_shape: tuple) -> np.ndarray`
  - Inverse reshape, then `[:H, :W]` to crop padding back.
- Edge case: 1-channel `np.ndarray` (grayscale) only — color is the 100% task.

### Task 4 — `dct_core/quant.py`
- Hard-code the standard JPEG **luminance** quantization table at Q=50 from `spec.md` (the chrominance table is for the 100% color task, so it can wait).
- `scale_table(base, q) -> np.ndarray`: use `5000/Q` for `Q<50`, `200-2*Q` for `Q≥50`, clamp to `[1, 255]`. Match the formula in `spec.md` exactly.
- `quantize(coeffs: np.ndarray, table: np.ndarray) -> np.ndarray`: `np.round(coeffs / table).astype(np.int32)`.
- `dequantize(qcoeffs: np.ndarray, table: np.ndarray) -> np.ndarray`: `qcoeffs.astype(np.float64) * table`.
- Constants: `LUM_TABLE_Q50` for use by `main.py`.

### Task 5 — `quality/metrics.py`
- `mse(a: np.ndarray, b: np.ndarray) -> float` — `float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))`.
- `psnr(a, b, max_val=255) -> float` — returns `float('inf')` if `mse == 0`; otherwise `10 * np.log10(max_val**2 / mse)`.
- `estimate_compressed_size(quantized_blocks: np.ndarray) -> int` — zig-zag scan each block, count `(run, value)` byte pairs (1 byte for run 0..255, 1 byte for value mapped from `[-127, 127]` to `[0, 254]`); return total bytes.
- `compression_ratio(orig_bytes: int, comp_bytes: int) -> float` — `orig_bytes / max(comp_bytes, 1)`.
- Zig-zag table is a standard `[0,1,5,2,3,4,6,7,...]` 8×8→64 permutation.

### Task 6 — `main.py` (50% MILESTONE)
- Use `argparse` to accept an optional `path` argument, default `assets/sample.png`.
- Friendly file-missing handling: if path doesn't exist, print
  ```
  No image found at '<path>'.
  Tip: run  python main.py path/to/your_photo.jpg
        (any PNG/JPG/BMP; color is auto-converted to grayscale)
  ```
  and `sys.exit(0)` — no traceback.
- Load via Pillow (`Image.open`), `.convert("L")`, then `np.asarray(img, dtype=np.float32)`.
- Pipeline (single function `compress_grayscale(img: np.ndarray, q: int = 50)`):
  1. Level shift: `img - 128`.
  2. `blocks, orig_shape, padded_shape = split_blocks(shifted)`.
  3. Vectorized DCT: `D @ blocks @ D.T` for all blocks at once using NumPy broadcasting on the last two axes.
  4. Build `table = scale_table(LUM_TABLE_Q50, q)`.
  5. `q_blocks = quantize(dct_blocks, table)`.
  6. `dct_blocks_back = dequantize(q_blocks, table)`.
  7. `reconstructed_shifted = idct2` via `D.T @ q_blocks_back @ D` for all blocks.
  8. Merge: `rec = merge_blocks(...) + 128`, `np.clip(rec, 0, 255).astype(np.uint8)`.
- Save `recon_50.png` next to the input path (i.e. in the input file's folder).
- Compute metrics vs original (`np.asarray(img, dtype=np.float32)`):
  - `original_bytes = img.size` (one byte per pixel — this is the "raw" baseline the spec compares to).
  - `compressed_bytes = estimate_compressed_size(q_blocks)`.
  - `cr = compression_ratio(original_bytes, compressed_bytes)`.
  - `mse_val = mse(orig, recon)`.
  - `psnr_val = psnr(orig, recon)`.
- Print aligned, easy-to-read summary to stdout:
  ```
  Image:           <W> x <H>
  Original size:   <KB> KB
  Compressed est.: <KB> KB
  Compression ratio: <CR>
  MSE:             <MSE>
  PSNR:            <PSNR> dB
  Saved reconstructed image to <out>
  ```

### Test File — `tests/test_dct.py`
- `test_orthogonality` — `D @ D.T ≈ 2 * I` within `1e-10`.
- `test_roundtrip` — random 8×8 block, `idct2(dct2(x)) ≈ x` within `1e-6`.
- `test_constant_block` — a block of 256 (mid-gray + level shift) has only `D[0,0]` non-zero, equal to value × 255 (closed-form check).
- No PNG needed for tests — purely in-memory NumPy blocks. This makes `pytest` independent of `assets/sample.png` and therefore verifiable during the brief.

### Documentation File — `PROGRESS.md`
- Created at the **beginning** of execution (before Task 2).
- After each of Tasks 1 → 6 completes, append a section:
  `## Task N — <title>` followed by 4 short paragraphs:
  1. **What we built** (file names + function names).
  2. **What it does / how it works** (one paragraph, no equations).
  3. **Where it fits in the pipeline** (one-sentence reference back to the spec diagram).
  4. **What's left** (one line pointing to the next task).
- Final section: **What the 50% project looks like tomorrow** — summary of running `python main.py assets/sample.png` and seeing the printed metrics + `recon_50.png`.

## Files Touched (summary)

| File | Action | Reason |
|------|--------|--------|
| `requirements.txt` | Create | Pin deps for the 50% run. |
| `dct_core/__init__.py` | Create empty | Package marker. |
| `dct_core/dct.py` | Create | DCT/IDCT core math (Task 2). |
| `dct_core/blocks.py` | Create | Block helpers (Task 3). |
| `dct_core/quant.py` | Create | Quantization (Task 4). |
| `quality/__init__.py` | Create empty | Package marker. |
| `quality/metrics.py` | Create | MSE, PSNR, CR, size estimate (Task 5). |
| `main.py` | Create | CLI demo (Task 6 / 50% milestone). |
| `tests/__init__.py` | Create empty | pytest package marker. |
| `tests/test_dct.py` | Create | DCT round-trip + orthogonality tests. |
| `assets/.gitkeep` | Create | Keep empty assets/ dir. |
| `PROGRESS.md` | Create, append after each task | Step-by-step easy-words log. |

No files outside the project root will be modified.

## Assumptions & Decisions
- **DCT normalization:** the spec uses the standard separable Type-II with `c(0)=1/√2`, `c(k>0)=1`. Implementation uses equivalent matrix form `D[k,i]=alpha(k)·cos((2i+1)kπ/16)` with `alpha(0)=1/√8`, `alpha(k>0)=1/2`. These are the same up to a row/column scalar convention; orthogonality `D·Dᵀ = 2I` confirms the choice.
- **Grayscale only at 50%:** the spec restricts the 50% milestone to grayscale, so color/YCCbCr is deferred.
- **Quantization scaling formula** uses `floor((base*scale + 50) / 100)` per the spec; we will use `np.clip(np.floor((base*scale + 50)/100), 1, 255).astype(np.int32)`.
- **Vectorized block DCT** uses NumPy broadcasting on the last two axes for performance; result is mathematically identical to a Python loop but ~100× faster on a 512×512 image.
- **Original size baseline** = raw pixel bytes (`W*H`), per the spec ("raw, uncompressed bytes per pixel"). This is the honest comparison for the compression ratio.
- **Sample image handling:** per user confirmation, no image is bundled. `main.py` is friendly if the file is missing. User must supply `assets/sample.png` (or any path) for the live demo.
- **CLI behavior on missing file:** print a friendly tip and `sys.exit(0)`. No traceback.

## Verification Steps
After all 6 tasks and `PROGRESS.md` are written:

1. `python -m pip install -r requirements.txt` (Windows PowerShell).
2. `python -c "import dct_core.dct, dct_core.quant, quality.metrics; print('imports ok')"` — smoke test.
3. `python -m pytest tests/ -v` — all DCT tests pass.
4. `python main.py` with no sample.png present — verify friendly message, exit 0, no traceback.
5. (Optional, requires user to drop in `assets/sample.png`) — `python main.py assets/sample.png` runs and prints the aligned summary block + saves `assets/recon_50.png`.
6. Update `tasks.md` checking Tasks 1–6 boxes.
7. Update `checklist.md` checking the 50% milestone boxes that are now verified (project skeleton exists, requirements file lists deps, modules expose the right names, `main.py` runs the missing-file friendly path, DCT tests pass).