# Progress Log — DCT Image Compression (50% Milestone)

This file is the easy-words diary of everything we built for the 50% demo.
It walks through the project step-by-step so you always know:

* what was just built,
* how it works in plain English,
* where it sits in the bigger DIP pipeline,
* what is left for the next 100% milestone.

The matching formal spec is in `.trae/specs/dct-image-compression/spec.md`.
The matching formal task list is in `.trae/specs/dct-image-compression/tasks.md`.

---

## What you can run today (the 50% demo)

```bash
# From the project root:
python main.py                          # uses assets/sample.png
python main.py path/to/your_photo.jpg   # any image, any path
python main.py assets/sample.png --q 10 # try Q = 1..100
```

What you see in the console is a small summary card like:

```
--------------------------------------------
  DCT Image Compression — 50% Milestone
--------------------------------------------
  Image:            256 x 256  (grayscale)
  Quality factor Q: 50
  Original size:    64.00 KB
  Compressed est.:  18.35 KB
  Compression ratio:   3.49 x
  MSE:               43.718
  PSNR:             31.72 dB
--------------------------------------------
  Saved reconstructed image to: ...\assets\sample_recon_q50.png
```

The reconstructed image is saved next to the input so you can flip between
the two and *see* the blocking artifacts that appear as Q drops.

If `assets/sample.png` is missing, the script prints a friendly tip instead
of crashing — no embarrassing traceback in front of the teacher.

---

## Task 1 — Project skeleton (DONE)

**What we built**

* `requirements.txt` pinning `numpy`, `Pillow`, `pytest`.
* Empty `__init__.py` files inside `dct_core/`, `quality/`, `tests/` so
  Python treats them as importable packages.
* `assets/.gitkeep` so the empty assets folder survives in version control.

**How it works / why it matters**

The skeleton just gives the project a clean home. A `requirements.txt` is the
modern way to say "here is exactly the software you need installed". The
`__init__.py` files turn folders into Python *packages*, which is what lets
`from dct_core.dct import dct2` work in `main.py` and the tests.

**Where it fits in the pipeline**

Task 1 is the empty lab bench. Tasks 2–6 are the experiments we put on it.

**What's left**

The math itself — DCT, IDCT, blocks, quantization, metrics.

---

## Task 2 — The 2-D DCT and inverse DCT (`dct_core/dct.py`)

**What we built**

A from-scratch implementation of the 2-D Type-II DCT (and its inverse) in
pure NumPy. The single function `dct2(block)` takes an 8×8 block of pixels
and returns an 8×8 block of frequency coefficients. `idct2(coeffs)` reverses
it. There are also batched versions `dct2_batch` / `idct2_batch` that handle
`(N, 8, 8)` stacks of blocks in one shot, which is what `main.py` uses.

**How it works in plain English**

We build a fixed 8×8 matrix `D` once at import time. Each row of `D` is a
"cosine wave" sampled 8 times across the block, and each row has a slightly
higher frequency than the row above it. Row 0 is the slowest (a flat
brightness), row 7 is the fastest wiggle.

To transform a block we do `D @ block @ D.T`. Think of this as: first,
compare the block against each row of `D` to see how much of every
horizontal frequency it contains (`D @ block`); then compare those results
against each column of `D` to see how much of every vertical frequency is
present (`@ D.T`). The output is a grid of numbers telling us "how much
slow-bright and how much fast-jittery is in this block".

To undo it we just flip the multiplication: `D.T @ coeffs @ D`. Because the
matrix `D` is constructed to be orthonormal, this perfectly recovers the
input up to floating-point noise (we tested it: error < 1×10⁻⁹).

**Where it fits in the pipeline**

This is the *transform* step in the spec diagram:

```
image → split 8x8 → DCT → quantize → dequantize → IDCT → reconstructed image
                          ^^^^^
                          you are here (and so is Task 4)
```

**What's left**

Wrapping the DCT into 8×8 blocks across a whole image, and turning the
coefficients into fewer bits via quantization.

---

## Task 3 — Splitting an image into 8×8 blocks (`dct_core/blocks.py`)

**What we built**

Two helpers:

* `split_blocks(img)` — takes a 2-D image and returns `(blocks,
  orig_shape, padded_shape)`. `blocks` is a `(N, 8, 8)` array ready for the
  batched DCT.
* `merge_blocks(blocks, orig_shape, padded_shape)` — the reverse:
  stitches the blocks back into an image and crops off any padding.

**How it works in plain English**

JPEG never DCTs the whole image at once. It cuts the image into small 8×8
tiles and DCTs each tile independently. We do the same. If the image is
500 pixels wide (not a multiple of 8), we *reflect-pad* the right edge —
mirror the last column a few times — so we get a clean multiple of 8. After
the inverse DCT we crop the padding back, so the user sees an image of the
exact original size.

Doing the math on small blocks is fast (4096 tiny matrix products on a
512×512 image) and is also why compression artifacts look like 8×8 squares
— each block is independent.

**Where it fits in the pipeline**

This is the "Divide image into 8×8 pixel blocks" arrow in the spec.

**What's left**

Throwing away information on purpose — quantization.

---

## Task 4 — Quantization with the JPEG luminance table (`dct_core/quant.py`)

**What we built**

The standard JPEG luminance quantization table at Q=50 (hard-coded exactly
as it appears in the JPEG standard), plus three functions:

* `scale_table(base, q)` — re-scales the table for any quality Q in 1..100.
  Q=100 → all 1s (lossless). Q=1 → huge numbers (very lossy).
* `quantize(coeffs, table)` — rounds each coefficient divided by the table
  entry to the nearest integer.
* `dequantize(qcoeffs, table)` — multiplies back by the table to recover an
  approximate coefficient.

**How it works in plain English**

After the DCT, each 8×8 block has one "average brightness" coefficient
(top-left, the DC) and 63 "detail" coefficients (the ACs, especially toward
the bottom-right). The eye is much less sensitive to fast-changing detail
than to slow-changing brightness, so we *deliberately throw away* precision
in those detail coefficients.

The quantization table is a recipe for how aggressively to throw away each
position. The top-left entry is small (preserve brightness well), and the
bottom-right entries are large (allow huge rounding error in fine detail).
That single table is the reason a 5 MB photo can shrink to ~700 KB and
still look fine.

When you change the Q slider in the GUI (the 100% milestone), this is the
table that changes underneath.

**Where it fits in the pipeline**

This is the "Quantization → Discard / reduce less important information"
arrows in the spec.

**What's left**

Now that we have a real (lossy) compressed representation, we need to
measure *how much* quality we lost.

---

## Task 5 — Quality metrics (`quality/metrics.py`)

**What we built**

Four small functions:

* `mse(a, b)` — average per-pixel squared difference between two images.
* `psnr(a, b)` — converts MSE to decibels. Bigger = better quality. Returns
  `inf` if the two images are identical (MSE is zero).
* `estimate_compressed_size(quantized_blocks)` — walks every block in
  zig-zag order, counts runs of zeros, and counts bytes as `(run, value)`
  pairs. More zeros → smaller estimate. This is what makes the demo's
  compression ratio *respond* to the Q slider.
* `compression_ratio(orig_bytes, comp_bytes)` — original / compressed.

**How it works in plain English**

MSE tells you "on average, how wrong is each pixel?" in raw squared units
of brightness. PSNR is the same idea on a logarithmic scale, in decibels —
that's the standard number you'll see in image-processing papers. A
perfect reconstruction is `inf` dB; ~30 dB looks fine to most people; ~20 dB
shows obvious artifacts.

The compression ratio is `original bytes / compressed bytes`. To make the
estimate fair we always compare against the *raw pixel size* (one byte per
pixel for grayscale), not the PNG/JPEG file size, because those formats
already do their own compression on top of ours.

**Where it fits in the pipeline**

This is the "Module 2 — Quality Analysis" block from the spec's project
overview.

**What's left**

Wiring all six pieces together into a runnable demo.

---

## Task 6 — The CLI demo (`main.py`) — **THIS IS THE 50% MILESTONE**

**What we built**

A single command-line script that runs the entire pipeline on a real
image and prints a clean summary card.

```
python main.py assets/sample.png          # default Q=50
python main.py my_photo.jpg --q 10       # very compressed
python main.py my_photo.jpg --q 90       # high quality
```

The script:

1. Loads the image with Pillow.
2. Converts to grayscale automatically (`.convert("L")`).
3. Runs the pipeline: level shift → split → DCT → quantize → dequantize →
   IDCT → inverse level shift → clip.
4. Saves the reconstructed image as `<name>_recon_q<Q>.png` next to the
   input.
5. Prints original size, estimated compressed size, compression ratio,
   MSE, PSNR, and the saved-file path.
6. If the file is missing, prints a friendly tip instead of a stack trace.

**How it works in plain English**

This is just glue code. It calls `load_grayscale`, then
`compress_grayscale` (which itself calls `split_blocks`, `dct2_batch`,
`scale_table`, `quantize`, `dequantize`, `idct2_batch`, `merge_blocks`),
then computes the metrics and prints them.

There is one extra design choice worth knowing: `quantize` and `dequantize`
now broadcast the 8×8 table against the `(N, 8, 8)` block stack, so a
single table is applied to every block automatically — that's why the
pipeline is so short.

**Where it fits in the pipeline**

This script is the "live demo" the spec promised: from raw image to
reconstructed image with measured numbers, in one command.

**What's left (for 100%)**

* Color support (RGB → YCbCr → per-channel pipeline).
* Tkinter GUI with a quality slider that updates the preview live.
* 4-up comparison grid (`Original | Q=90 | Q=50 | Q=10`).
* Benchmark sweep across Q values → CSV.
* Polished README and project report.

The underlying math does not change in 100% — it's the same DCT, the same
quantization, the same metrics. We just add a colour converter, a GUI, and
a benchmark script.

---

## What the 50% project looks like tomorrow

If everything is in place:

1. You put a real photo (faces, texture, anything with detail) in
   `assets/sample.png`.
2. You open a terminal in the project folder.
3. You run `python main.py`.
4. The script prints the summary card and saves `assets/sample_recon_q50.png`.
5. You open both images and point at the differences: at Q=50 the picture
   is almost the same; at Q=10 (`python main.py assets/sample.png --q 10`)
   you can clearly see 8×8 blocking.
6. You say:

> "Our project demonstrates how DCT can be used for lossy image
> compression. We transform the image into frequency components, quantize
> them to reduce the amount of data, reconstruct the image, and then
> analyze the trade-off between compression and image quality using MSE,
> PSNR, and compression ratio. JPEG is a real-world example of this type
> of image-compression approach."

You have, demonstrably:

* Implemented the entire signal-processing chain from scratch.
* Used the same luminance quantization table that JPEG uses.
* Measured, not guessed, the trade-off between size and quality.

That is the 50% milestone. Everything after that is polish and tooling.
