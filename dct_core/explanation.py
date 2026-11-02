"""
dct_core.explanation
====================
DIP Course Project Educational Core:
Algorithms, Mathematical Formulas, Pipeline Flowcharts, and Live Numerical
Calculations for DCT-based Image Compression.

Designed for course evaluation, viva presentation, and academic demonstration.
Provides:
  1. Complete ASCII / Unicode Pipeline Flowcharts.
  2. Formal Algorithms with input/output, complexity, and pseudocode.
  3. Mathematical Formulas reference with exact DIP definitions.
  4. Live Numerical Calculation Walkthrough for an 8x8 block (Level shift ->
     DCT -> Quantization -> Zig-zag/RLE -> Dequantization -> IDCT -> Metrics).
  5. Global Image Metrics breakdown and derivations (MSE, PSNR, CR, BPP).
  6. Exportable academic report generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dct import BLOCK, D, dct2, idct2
from .quant import LUM_TABLE_Q50, dequantize, quantize, scale_table
from quality.metrics import _ZIGZAG, compression_ratio, mse, psnr


# Classic 8x8 luminance test block from Digital Image Processing (Gonzalez & Woods)
STANDARD_SAMPLE_BLOCK = np.array(
    [
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 55, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94],
    ],
    dtype=np.uint8,
)


@dataclass
class BlockCalculationSteps:
    """Stores exact numerical matrices and intermediate values for an 8x8 block."""
    quality: int
    scale_factor: float
    original: np.ndarray             # (8, 8) uint8 [0, 255]
    shifted: np.ndarray              # (8, 8) float64 [-128, +127]
    dct_coeffs: np.ndarray           # (8, 8) float64
    dc_coeff: float                  # F(0, 0)
    total_energy: float              # sum(F^2)
    dc_energy_pct: float             # (F(0,0)^2 / total_energy) * 100
    top3x3_energy_pct: float         # energy in low-frequency 3x3 sub-block
    quant_table: np.ndarray          # (8, 8) int32 scaled quantization table
    quant_coeffs: np.ndarray         # (8, 8) int32 quantized integers
    zeros_count: int                 # number of zero coefficients out of 64
    zeros_pct: float                 # percentage of zero coefficients
    zigzag_array: np.ndarray         # 1D array of 64 elements in zig-zag order
    rle_pairs: List[Tuple[int, int]] # list of (run_of_zeros, non_zero_val)
    block_bytes: int                 # estimated byte size for this block
    dequant_coeffs: np.ndarray       # (8, 8) float64 dequantized coefficients
    quant_error: np.ndarray          # (8, 8) float64 (dct_coeffs - dequant_coeffs)
    idct_shifted: np.ndarray         # (8, 8) float64 reconstructed shifted block
    reconstructed: np.ndarray        # (8, 8) uint8 reconstructed pixels in [0, 255]
    spatial_error: np.ndarray        # (8, 8) float64 (original - reconstructed)
    block_mse: float                 # mean squared error on this block
    block_psnr: float                # PSNR (dB) on this block


def format_matrix(
    mat: np.ndarray,
    fmt: str = "{:6.1f}",
    prefix: str = "    ",
    header_indices: bool = True,
) -> str:
    """Format an 8x8 numpy array into a clean monospace ASCII grid."""
    lines = []
    if header_indices:
        cols_hdr = prefix + "    " + "".join(f"[{c:^5}]" for c in range(8))
        lines.append(cols_hdr)
        lines.append(prefix + "    " + "-" * 56)

    for r in range(mat.shape[0]):
        row_str = "".join(fmt.format(mat[r, c]) for c in range(mat.shape[1]))
        if header_indices:
            lines.append(f"{prefix}[{r}] {row_str}")
        else:
            lines.append(f"{prefix}{row_str}")
    return "\n".join(lines)


def calculate_block_steps(
    block: np.ndarray,
    quality: int = 75,
) -> BlockCalculationSteps:
    """Compute and return every numerical intermediate in the 8x8 block pipeline.

    Parameters:
        block: (8, 8) array of pixel values in [0, 255].
        quality: JPEG quality factor Q in [1, 100].

    Returns:
        BlockCalculationSteps with all numerical matrices and metrics.
    """
    if block.shape != (BLOCK, BLOCK):
        raise ValueError(f"calculate_block_steps expects an (8, 8) block, got {block.shape}")
    if not (1 <= quality <= 100):
        raise ValueError(f"Quality must be in [1, 100], got {quality}")

    orig_u8 = np.clip(block, 0, 255).astype(np.uint8)

    # 1. Level shift: f(x, y) - 128
    shifted = orig_u8.astype(np.float64) - 128.0

    # 2. Forward 2-D DCT: F = D @ shifted @ D.T
    coeffs = dct2(shifted)
    dc_val = float(coeffs[0, 0])

    # Energy calculations (Parseval's theorem in orthogonal transforms)
    sq_coeffs = coeffs ** 2
    total_energy = float(np.sum(sq_coeffs))
    dc_energy = float(coeffs[0, 0] ** 2)
    dc_energy_pct = (dc_energy / total_energy * 100.0) if total_energy > 0 else 100.0
    top3x3_energy = float(np.sum(sq_coeffs[:3, :3]))
    top3x3_energy_pct = (top3x3_energy / total_energy * 100.0) if total_energy > 0 else 100.0

    # 3. Quantization Table Scaling
    q_scale = 5000 / quality if quality < 50 else (200 - 2 * quality)
    q_table = scale_table(LUM_TABLE_Q50, quality)

    # 4. Quantization: F_Q = round( F / q_table )
    q_coeffs = quantize(coeffs, q_table)
    zeros_count = int(np.sum(q_coeffs == 0))
    zeros_pct = (zeros_count / 64.0) * 100.0

    # 5. Zig-zag order traversal
    flat_q = q_coeffs.flatten()
    zigzag_seq = flat_q[_ZIGZAG]

    # 6. Run-Length Encoding (RLE) simulation
    rle_pairs: List[Tuple[int, int]] = []
    run = 0
    total_bytes = 0
    for val in zigzag_seq:
        if val == 0:
            run += 1
            if run > 255:
                rle_pairs.append((255, 0))
                run = 0
            continue
        rle_pairs.append((run, int(val)))
        total_bytes += 2  # (run, value) pair
        run = 0
    total_bytes += 1  # End-Of-Block (EOB) marker
    block_bytes = total_bytes

    # 7. Dequantization: F'_Q = F_Q * q_table
    dequant_coeffs = dequantize(q_coeffs, q_table)
    quant_error = coeffs - dequant_coeffs

    # 8. Inverse 2-D DCT: f'_shifted = D.T @ dequant_coeffs @ D
    rec_shifted = idct2(dequant_coeffs)

    # 9. Inverse level shift: f' = clip(f'_shifted + 128, 0, 255)
    rec_u8 = np.clip(np.round(rec_shifted + 128.0), 0, 255).astype(np.uint8)

    # 10. Spatial error & quality metrics
    spat_err = orig_u8.astype(np.float64) - rec_u8.astype(np.float64)
    b_mse = float(np.mean(spat_err ** 2))
    b_psnr = psnr(orig_u8.astype(np.float64), rec_u8.astype(np.float64))

    return BlockCalculationSteps(
        quality=quality,
        scale_factor=float(q_scale),
        original=orig_u8,
        shifted=shifted,
        dct_coeffs=coeffs,
        dc_coeff=dc_val,
        total_energy=total_energy,
        dc_energy_pct=dc_energy_pct,
        top3x3_energy_pct=top3x3_energy_pct,
        quant_table=q_table,
        quant_coeffs=q_coeffs,
        zeros_count=zeros_count,
        zeros_pct=zeros_pct,
        zigzag_array=zigzag_seq,
        rle_pairs=rle_pairs,
        block_bytes=block_bytes,
        dequant_coeffs=dequant_coeffs,
        quant_error=quant_error,
        idct_shifted=rec_shifted,
        reconstructed=rec_u8,
        spatial_error=spat_err,
        block_mse=b_mse,
        block_psnr=b_psnr,
    )


def get_pipeline_flowchart() -> str:
    """Return the structured ASCII/Unicode pipeline flowchart."""
    return """================================================================================
          DIGITAL IMAGE PROCESSING (DIP) -- DCT IMAGE COMPRESSION PIPELINE
================================================================================

                             [ INPUT IMAGE ]
                 (H x W Grayscale  OR  H x W x 3 RGB Color)
                                    |
                                    v
                       < Is image RGB Color? >
                        /                    \\
                     YES                      NO
                      |                        |
                      v                        v
            +---------------------+   +---------------------+
            | COLOR DECOMPOSITION |   | DIRECT LUMA CHANNEL |
            | RGB -> YCbCr Space  |   | Grayscale Y = I     |
            | (ITU-R BT.601)      |   +---------------------+
            +---------------------+              |
               |         |                       |
               | Y       | Cb, Cr                |
               v         v                       |
         [Quality Q]  [Quality Q//2]             |
               \\         /                       |
                v       v                        |
            +------------------------------------+
            |
            v
+==============================================================================+
| ENCODER (COMPRESSION STAGE)                                                  |
+==============================================================================+
|                                                                              |
|  1. LEVEL SHIFTING:                                                          |
|     f_shift(x, y) = f(x, y) - 128                                            |
|     Centers unsigned [0, 255] pixels to signed zero-mean range [-128, +127]. |
|     Prevents massive DC bias and numerical overflow during transform.        |
|                                                                              |
|                               |                                              |
|                               v                                              |
|  2. BLOCK PARTITIONING (8x8):                                                |
|     Divides channel into N non-overlapping 8x8 spatial tiles.                |
|     Edge padding (reflection mode) applied if H or W not divisible by 8.     |
|                                                                              |
|                               |                                              |
|                               v                                              |
|  3. FORWARD 2-D DISCRETE COSINE TRANSFORM (DCT-II):                          |
|     F = D @ f_shift @ D^T                                                    |
|     Energy Compaction: Converts spatial pixel correlations into uncorrelated |
|     frequency coefficients. Concentrates ~90-98% signal energy into top-left.|
|                                                                              |
|                               |                                              |
|                               v                                              |
|  4. PSYCHOVISUAL QUANTIZATION (LOSSY CORE):                                  |
|     Q_scaled = scale_table(Q50_table, Q)                                     |
|     F_Q(u, v) = round( F(u, v) / Q_scaled(u, v) )                            |
|     Discards high-frequency details invisible to human visual system (HVS).  |
|     Turns vast majority of AC coefficients to EXACT ZEROS.                   |
|                                                                              |
|                               |                                              |
|                               v                                              |
|  5. ZIG-ZAG SCAN & RUN-LENGTH ENCODING (ENTROPY REDUCTION):                  |
|     Reorders 8x8 matrix into 1D array from lowest (DC) to highest AC freq.   |
|     Compresses consecutive zero-runs into compact (run, value) pairs + EOB.  |
|                                                                              |
+==============================================================================+
                                |
                                | [Compressed Bitstream / Encoded Coefficients]
                                v
+==============================================================================+
| DECODER (RECONSTRUCTION STAGE)                                               |
+==============================================================================+
|                                                                              |
|  6. DEQUANTIZATION (INVERSE QUANTIZATION):                                   |
|     F'_Q(u, v) = F_Q(u, v) * Q_scaled(u, v)                                  |
|     Restores approximate DCT spectral coefficients.                          |
|     Quantization noise: e(u, v) = F(u, v) - F'_Q(u, v).                      |
|                                                                              |
|                               |                                              |
|                               v                                              |
|  7. INVERSE 2-D DISCRETE COSINE TRANSFORM (IDCT-III):                        |
|     f'_shift = D^T @ F'_Q @ D                                                |
|     Synthesizes spatial domain block from reconstructed frequencies.         |
|                                                                              |
|                               |                                              |
|                               v                                              |
|  8. INVERSE LEVEL SHIFT & CLIPPING:                                          |
|     f'(x, y) = clip( round(f'_shift(x, y) + 128), 0, 255 )                   |
|     Restores unsigned [0..255] range and clamps out-of-bounds ringing.       |
|                                                                              |
|                               |                                              |
|                               v                                              |
|  9. BLOCK STITCHING & CROPPING:                                              |
|     Assembles (N, 8, 8) blocks into full channel grid and crops edge padding.|
|                                                                              |
|                               |                                              |
|                               v                                              |
| 10. COLOR RECONSTRUCTION (IF COLOR):                                         |
|     YCbCr -> RGB via inverse BT.601 matrix.                                  |
|                                                                              |
+==============================================================================+
                                |
                                v
                  [ RECONSTRUCTED OUTPUT IMAGE ]
                                |
                                v
               [ OBJECTIVE QUALITY EVALUATION ]
          * Mean Squared Error (MSE)
          * Peak Signal-to-Noise Ratio (PSNR in dB)
          * Compression Ratio (CR:1)
          * Bits Per Pixel (BPP)
================================================================================"""


def get_algorithms_text() -> str:
    """Return the formal DIP algorithms with pseudocode and explanations."""
    return """================================================================================
               FORMAL DIP ALGORITHMS & IMPLEMENTATION DETAILS
================================================================================

ALGORITHM 1: FORWARD 2-D DISCRETE COSINE TRANSFORM (DCT-II)
--------------------------------------------------------------------------------
Input:       8x8 Level-shifted spatial block f_shift[x, y] in range [-128, +127].
Output:      8x8 Matrix of frequency coefficients F[u, v].
Properties:  Linear, Orthonormal, Separable.
Complexity:  O(N^3) using matrix factorization (vs O(N^4) naive summation).

Pseudocode:
1. Precompute 1-D orthonormal DCT basis matrix D (8x8):
       For k = 0 to 7:
           alpha(k) = 1/sqrt(8) if k == 0 else sqrt(2/8) = 0.5
           For i = 0 to 7:
               D[k, i] = alpha(k) * cos( (2*i + 1) * k * pi / 16 )
2. Perform separable row-column transform via matrix multiplications:
       F = D * f_shift * D^T
3. F[0, 0] is the DC coefficient (reflects average block intensity).
   F[u, v] for (u, v) != (0, 0) are AC coefficients (directional frequencies).


ALGORITHM 2: ADAPTIVE QUANTIZATION & QUALITY FACTOR SCALING
--------------------------------------------------------------------------------
Input:       DCT coefficient block F[u, v], Quality factor Q in [1, 100],
             Standard Base Luminance Table Q50[u, v].
Output:      Quantized integer matrix F_Q[u, v], Scaled table Q_scaled[u, v].
Principle:   Human Visual System (HVS) is sensitive to low spatial frequencies
             and insensitive to high-frequency diagonal textures.

Pseudocode (Standard ISO / libjpeg Scaling Formula):
1. Compute scale multiplier S based on Quality Q:
       if Q < 50:
           S = 5000 / Q
       else:
           S = 200 - 2 * Q
2. Scale each base table entry and clamp to [1, 255]:
       For u = 0 to 7, v = 0 to 7:
           val = floor( (Q50[u, v] * S + 50) / 100 )
           Q_scaled[u, v] = min( 255, max( 1, val ) )
3. Quantize coefficients:
       F_Q[u, v] = round( F[u, v] / Q_scaled[u, v] )
   Note: This rounding step is the SOLE lossy operation in the entire system.


ALGORITHM 3: ZIG-ZAG SCANNING & RUN-LENGTH ENCODING (RLE)
--------------------------------------------------------------------------------
Input:       Quantized 8x8 integer matrix F_Q[u, v].
Output:      1D serialized stream of (run_zeros, non_zero_value) pairs + EOB.
Rationale:   Low frequencies cluster in the top-left; high frequencies cluster
             in the bottom-right. Zig-zag order guarantees that high-frequency
             zeros are grouped into a long terminal sequence.

Pseudocode:
1. Traverse matrix indices along diagonals:
   (0,0) -> (0,1) -> (1,0) -> (2,0) -> (1,1) -> (0,2) -> (0,3) -> ... -> (7,7).
2. Initialize run_counter = 0, byte_count = 0.
3. For each coefficient val in zig-zag order:
       if val == 0:
           run_counter += 1
           if run_counter > 255:
               emit (255, 0); run_counter = 0
       else:
           emit (run_counter, val)
           byte_count += 2  # 1 byte for run length, 1 byte for value magnitude
           run_counter = 0
4. Emit End-Of-Block (EOB = 0x00) marker; byte_count += 1.


ALGORITHM 4: DEQUANTIZATION & 2-D INVERSE DCT (IDCT-III)
--------------------------------------------------------------------------------
Input:       Quantized block F_Q[u, v], Scaled table Q_scaled[u, v], Basis D.
Output:      Reconstructed uint8 spatial block f_recon[x, y].

Pseudocode:
1. Dequantize (Scalar Multiplication):
       F'_Q[u, v] = F_Q[u, v] * Q_scaled[u, v]
2. 2-D Inverse DCT via Transpose Basis:
       Because D is orthonormal (D * D^T = I), the inverse is exact transpose:
       f'_shift = D^T * F'_Q * D
3. Inverse Level Shift and Boundary Clamping:
       f_recon[x, y] = clip( round( f'_shift[x, y] + 128 ), 0, 255 )


ALGORITHM 5: FULL-RANGE COLOR SPACE CONVERSION (BT.601)
--------------------------------------------------------------------------------
Input:       RGB uint8 pixel [R, G, B] in [0, 255].
Output:      YCbCr pixel [Y, Cb, Cr] in [0, 255].

Forward Transformation Matrix (Full-Range):
    | Y  |   |  0.299000   0.587000   0.114000 |   | R |   |   0 |
    | Cb | = | -0.168736  -0.331264   0.500000 | * | G | + | 128 |
    | Cr |   |  0.500000  -0.418688  -0.081312 |   | B |   | 128 |

Inverse Transformation Matrix:
    | R |   | 1.000000   0.000000   1.402000 |   | Y       |
    | G | = | 1.000000  -0.344136  -0.714136 | * | Cb - 128 |
    | B |   | 1.000000   1.772000   0.000000 |   | Cr - 128 |
================================================================================"""


def get_formulas_text() -> str:
    """Return comprehensive mathematical formulas reference with symbol explanations."""
    return """================================================================================
                    MATHEMATICAL FORMULAS REFERENCE
================================================================================

1. 2-D DISCRETE COSINE TRANSFORM (TYPE-II)
--------------------------------------------------------------------------------
Continuous mathematical summation over an N x N block (N = 8):

                7     7
  F(u, v) = a(u) a(v) SUM   SUM  f_shift(x, y) * cos[(2x+1)u*pi / 16] * cos[(2y+1)v*pi / 16]
               x=0   y=0

Where the orthonormal normalization scale factor a(k) is:
             /
             |  1 / sqrt(8)        if k = 0   (DC component)
     a(k) = <
             |  sqrt(2 / 8) = 0.5  if k > 0   (AC components)
             \\

Matrix Formulation (Separable Form):
     F = D * f_shift * D^T
Where D is the 8x8 matrix:
     D[k, i] = a(k) * cos( (2i + 1) * k * pi / 16 )
Orthonormality Identity:
     D * D^T = I_8   and   D^T * D = I_8


2. 2-D INVERSE DISCRETE COSINE TRANSFORM (TYPE-III / IDCT)
--------------------------------------------------------------------------------
Summation:
               7     7
  f(x, y) =   SUM   SUM  a(u) a(v) F(u, v) * cos[(2x+1)u*pi / 16] * cos[(2y+1)v*pi / 16]
              u=0   v=0

Matrix Formulation:
     f_shift = D^T * F * D


3. STANDARD JPEG LUMINANCE QUANTIZATION TABLE (Q = 50)
--------------------------------------------------------------------------------
         v=0   v=1   v=2   v=3   v=4   v=5   v=6   v=7
  u=0 [   16    11    10    16    24    40    51    61  ]
  u=1 [   12    12    14    19    26    58    60    55  ]
  u=2 [   14    13    16    24    40    57    69    56  ]
  u=3 [   14    17    22    29    51    87    80    62  ]
  u=4 [   18    22    37    56    68   103    77    62  ]
  u=5 [   30    35    55    64    81   104    92    75  ]
  u=6 [   50    72    78    87   103   121   100    78  ]
  u=7 [   68    92    95    98   112   100   103    99  ]


4. JPEG QUALITY FACTOR SCALING EQUATION (ISO/IEC IS 10918-1)
--------------------------------------------------------------------------------
Given user quality Q in [1, 100]:
             /
             |  5000 / Q        if Q < 50   (coarser quantization, high compression)
     S(Q) = <
             |  200 - 2 * Q     if Q >= 50  (finer quantization, high quality)
             \\

  Q_scaled(u, v) = clip( floor( (Q50(u, v) * S(Q) + 50) / 100 ), 1, 255 )

Forward Quantization:
  F_Q(u, v) = round( F(u, v) / Q_scaled(u, v) )

Dequantization (Reconstruction):
  F'_Q(u, v) = F_Q(u, v) * Q_scaled(u, v)

Quantization Distortion Error:
  e_Q(u, v) = F(u, v) - F'_Q(u, v)


5. STANDARD ZIG-ZAG TRAVERSAL MATRIX
--------------------------------------------------------------------------------
Maps 2-D coordinates (u, v) to 1-D sequence index (0 to 63):
      0   1   5   6  14  15  27  28
      2   4   7  13  16  26  29  42
      3   8  12  17  25  30  41  43
      9  11  18  24  31  40  44  53
     10  19  23  32  39  45  52  54
     20  22  33  38  46  51  55  60
     21  34  37  47  50  56  59  61
     35  36  48  49  57  58  62  63


6. OBJECTIVE IMAGE QUALITY & COMPRESSION METRICS
--------------------------------------------------------------------------------
A. Mean Squared Error (MSE):
                      1        M-1   N-1   C-1
     MSE = ------------------  SUM   SUM   SUM  [ I(x, y, c) - I_recon(x, y, c) ]^2
            M * N * C          x=0   y=0   c=0

B. Peak Signal-to-Noise Ratio (PSNR):
                         /    MAX_I^2   \\                    /      255      \\
     PSNR = 10 * log_10 |  ------------- | = 20 * log_10 | -------------- |  (dB)
                         \\      MSE     /                \\   sqrt(MSE)   /
     * If MSE = 0 (perfect reconstruction), PSNR = +infinity dB.
     * Typical acceptable quality threshold: > 30 dB.

C. Compression Ratio (CR):
            Uncompressed Raw Size (Bytes)         M * N * C * 1 Byte
     CR = --------------------------------- = --------------------------
             Compressed Bitstream (Bytes)          Total Encoded Bytes

D. Bits Per Pixel (BPP):
            Total Encoded Bytes * 8 bits
    BPP = -------------------------------  (bits/pixel)
                      M * N

E. Space Savings Percentage:
    Savings = ( 1 - 1 / CR ) * 100 %
================================================================================"""


def format_block_walkthrough(steps: BlockCalculationSteps) -> str:
    """Format the complete numerical step-by-step calculation report for an 8x8 block."""
    out = []
    out.append("================================================================================")
    out.append(f"       LIVE NUMERICAL CALCULATION WALKTHROUGH -- 8x8 BLOCK (QUALITY Q = {steps.quality})")
    out.append("================================================================================\n")

    # Step 1
    out.append("STEP 1: ORIGINAL 8x8 PIXEL BLOCK f(x, y)  [Range: 0 to 255 unsigned]")
    out.append("-" * 80)
    out.append(format_matrix(steps.original, fmt="{:5d} "))
    out.append("")

    # Step 2
    out.append("STEP 2: LEVEL-SHIFTED BLOCK f_shift(x, y) = f(x, y) - 128  [Range: -128 to +127]")
    out.append("-" * 80)
    out.append("Reason: Centers pixel intensities around zero to minimize DC coefficient bias.")
    out.append(format_matrix(steps.shifted, fmt="{:6.0f}"))
    out.append("")

    # Step 3
    out.append("STEP 3: FORWARD 2-D DCT COEFFICIENT MATRIX F(u, v) = D * f_shift * D^T")
    out.append("-" * 80)
    out.append(f"  * DC Coefficient F(0, 0) = {steps.dc_coeff:.2f}")
    out.append(f"  * Total Block Energy     = {steps.total_energy:.2f}")
    out.append(f"  * DC Energy Percentage   = {steps.dc_energy_pct:.2f}% of total block energy")
    out.append(f"  * Top-Left 3x3 Energy    = {steps.top3x3_energy_pct:.2f}% of total block energy")
    out.append("  (Notice how almost all signal power is compacted into the top-left frequencies!)\n")
    out.append(format_matrix(steps.dct_coeffs, fmt="{:7.1f}"))
    out.append("")

    # Step 4
    out.append(f"STEP 4: SCALED QUANTIZATION TABLE Q_scaled(u, v)  [Q = {steps.quality}, Scale Factor = {steps.scale_factor:.2f}]")
    out.append("-" * 80)
    out.append(f"Formula: S = {'5000 / Q' if steps.quality < 50 else '200 - 2*Q'} = {steps.scale_factor:.2f}")
    out.append("Q_scaled(u, v) = clip( floor( (Q50(u, v) * S + 50) / 100 ), 1, 255 )\n")
    out.append(format_matrix(steps.quant_table, fmt="{:5d} "))
    out.append("")

    # Step 5
    out.append("STEP 5: QUANTIZED INTEGER COEFFICIENTS F_Q(u, v) = round( F(u, v) / Q_scaled(u, v) )")
    out.append("-" * 80)
    sample_calc = (
        f"Sample Calculation:\n"
        f"  DC term: F(0, 0) / Q_s(0, 0) = {steps.dct_coeffs[0, 0]:.1f} / {steps.quant_table[0, 0]} = "
        f"{steps.dct_coeffs[0, 0] / steps.quant_table[0, 0]:.2f} -> round -> {steps.quant_coeffs[0, 0]}\n"
        f"  AC term: F(0, 1) / Q_s(0, 1) = {steps.dct_coeffs[0, 1]:.1f} / {steps.quant_table[0, 1]} = "
        f"{steps.dct_coeffs[0, 1] / steps.quant_table[0, 1]:.2f} -> round -> {steps.quant_coeffs[0, 1]}"
    )
    out.append(sample_calc)
    out.append(f"  * ZERO COEFFICIENTS: {steps.zeros_count} / 64 ({steps.zeros_pct:.1f}% Sparsity!)")
    out.append("  (These zeros require almost no storage, which creates the image compression!)\n")
    out.append(format_matrix(steps.quant_coeffs, fmt="{:5d} "))
    out.append("")

    # Step 6
    out.append("STEP 6: ZIG-ZAG SCAN & RUN-LENGTH ENCODING (RLE)")
    out.append("-" * 80)
    out.append("1D Zig-Zag Traversal Sequence (64 elements):")
    zz_str = "  " + ", ".join(str(int(v)) for v in steps.zigzag_array[:32]) + ",\n  "
    zz_str += ", ".join(str(int(v)) for v in steps.zigzag_array[32:])
    out.append(zz_str)
    out.append("\nRun-Length Pairs (run_of_zeros, non_zero_value):")
    pairs_str = "  " + ", ".join(f"({r}, {v})" for r, v in steps.rle_pairs)
    out.append(pairs_str if steps.rle_pairs else "  (All AC coefficients are zero!)")
    out.append(f"  + End-Of-Block (EOB = 0x00) marker")
    out.append(f"Estimated Block Encoded Size: {steps.block_bytes} bytes (Raw block = 64 bytes -> CR = {64.0 / steps.block_bytes:.2f}x)")
    out.append("")

    # Step 7
    out.append("STEP 7: DEQUANTIZATION F'_Q(u, v) = F_Q(u, v) * Q_scaled(u, v)")
    out.append("-" * 80)
    out.append(format_matrix(steps.dequant_coeffs, fmt="{:7.1f}"))
    out.append("")

    # Step 8
    out.append("STEP 8: RECONSTRUCTED PIXELS f_recon(x, y) = clip( round( D^T * F'_Q * D + 128 ), 0, 255 )")
    out.append("-" * 80)
    out.append(format_matrix(steps.reconstructed, fmt="{:5d} "))
    out.append("")

    # Step 9
    out.append("STEP 9: RESIDUAL ERROR MATRIX Delta(x, y) = f(x, y) - f_recon(x, y)")
    out.append("-" * 80)
    out.append(format_matrix(steps.spatial_error, fmt="{:5.0f} "))
    psnr_str = "inf dB" if steps.block_psnr == float("inf") else f"{steps.block_psnr:.2f} dB"
    out.append(f"\nLOCAL BLOCK QUALITY METRICS:")
    out.append(f"  * Mean Squared Error (MSE)          = {steps.block_mse:.3f}")
    out.append(f"  * Root Mean Squared Error (RMSE)    = {np.sqrt(steps.block_mse):.3f}")
    out.append(f"  * Peak Signal-to-Noise Ratio (PSNR) = {psnr_str}")
    out.append("================================================================================")

    return "\n".join(out)


def get_full_image_metrics_breakdown(
    original_u8: np.ndarray,
    reconstructed_u8: np.ndarray,
    compressed_bytes: int,
    quality: int,
    is_color: bool,
) -> str:
    """Generate detailed mathematical step-by-step breakdown of global image metrics."""
    h, w = original_u8.shape[:2]
    c = original_u8.shape[2] if original_u8.ndim == 3 else 1
    total_pixels = h * w
    total_samples = total_pixels * c
    raw_bytes = total_samples

    blocks_h = (h + BLOCK - 1) // BLOCK
    blocks_w = (w + BLOCK - 1) // BLOCK
    total_blocks_per_channel = blocks_h * blocks_w
    total_blocks = total_blocks_per_channel * c

    diff = original_u8.astype(np.float64) - reconstructed_u8.astype(np.float64)
    sse = float(np.sum(diff * diff))
    calc_mse = sse / total_samples
    calc_psnr = float("inf") if calc_mse <= 0 else 10.0 * np.log10((255.0 ** 2) / calc_mse)
    cr = float(raw_bytes) / float(compressed_bytes) if compressed_bytes > 0 else float("inf")
    bpp = (compressed_bytes * 8.0) / total_pixels
    space_savings = (1.0 - 1.0 / cr) * 100.0 if cr > 0 else 0.0

    psnr_disp = "+infinity dB" if calc_psnr == float("inf") else f"{calc_psnr:.2f} dB"

    lines = [
        "================================================================================",
        "          GLOBAL IMAGE COMPRESSION METRICS & STEP-BY-STEP DERIVATION",
        "================================================================================",
        f"IMAGE GEOMETRY & BLOCK PARTITIONING:",
        f"  * Dimensions:            {w} (Width) x {h} (Height) pixels",
        f"  * Color Mode:            {'3-Channel RGB (Color)' if is_color else '1-Channel Grayscale'}",
        f"  * Total Pixel Count:     {total_pixels:,} pixels",
        f"  * Total Pixel Samples:   {total_samples:,} scalar values (W x H x C)",
        f"  * 8x8 Grid Dimensions:   {blocks_w} blocks wide x {blocks_h} blocks high",
        f"  * Total 8x8 Blocks:      {total_blocks:,} blocks ({total_blocks_per_channel:,} per channel)",
        f"  * Quality Factor (Q):    {quality} (Luma Q={quality}, Chroma Q={max(1, quality // 2) if is_color else quality})",
        "",
        f"STORAGE & COMPRESSION DERIVATION:",
        f"  1. Raw Uncompressed Size:",
        f"     Size_raw = Width * Height * Channels * 1 Byte/sample",
        f"              = {w} * {h} * {c} = {raw_bytes:,} Bytes ({raw_bytes / 1024.0:.2f} KB)",
        "",
        f"  2. Compressed Size (Measured Zig-zag + RLE + EOB):",
        f"     Size_compressed = {compressed_bytes:,} Bytes ({compressed_bytes / 1024.0:.2f} KB)",
        "",
        f"  3. Compression Ratio (CR):",
        f"     CR = Size_raw / Size_compressed",
        f"        = {raw_bytes:,} / {compressed_bytes:,} = {cr:.2f} : 1  ({cr:.2f}x reduction)",
        "",
        f"  4. Bits Per Pixel (BPP):",
        f"     BPP = (Size_compressed * 8 bits) / (Width * Height)",
        f"         = ({compressed_bytes:,} * 8) / {total_pixels:,} = {bpp:.3f} bits/pixel",
        f"     (Raw baseline was {c * 8} bits/pixel -> {((c * 8 - bpp) / (c * 8)) * 100:.1f}% bitrate reduction)",
        "",
        f"  5. Space Savings Percentage:",
        f"     Savings = (1 - 1 / CR) * 100% = {space_savings:.2f}%",
        "",
        f"FIDELITY & ERROR METRICS DERIVATION:",
        f"  1. Sum of Squared Errors (SSE):",
        f"     SSE = SUM_x SUM_y SUM_c [ I(x,y,c) - I_recon(x,y,c) ]^2 = {sse:,.1f}",
        "",
        f"  2. Mean Squared Error (MSE):",
        f"     MSE = SSE / (Width * Height * Channels)",
        f"         = {sse:,.1f} / {total_samples:,} = {calc_mse:.4f}",
        "",
        f"  3. Root Mean Squared Error (RMSE):",
        f"     RMSE = sqrt(MSE) = sqrt({calc_mse:.4f}) = {np.sqrt(calc_mse):.4f} intensity levels",
        "",
        f"  4. Peak Signal-to-Noise Ratio (PSNR):",
        f"     PSNR = 10 * log_10( 255^2 / MSE )",
        f"          = 10 * log_10( 65025 / {calc_mse:.4f} ) = {psnr_disp}",
        "================================================================================",
    ]
    return "\n".join(lines)


def generate_full_academic_report(
    original_u8: np.ndarray,
    reconstructed_u8: np.ndarray,
    compressed_bytes: int,
    quality: int,
    is_color: bool,
    block_steps: BlockCalculationSteps,
) -> str:
    """Generate complete all-in-one text report suitable for course submission."""
    sections = [
        "================================================================================",
        "  DIGITAL IMAGE PROCESSING (DIP) COURSE PROJECT: DCT IMAGE COMPRESSION REPORT",
        "================================================================================\n",
        get_pipeline_flowchart(),
        "\n\n",
        get_algorithms_text(),
        "\n\n",
        get_formulas_text(),
        "\n\n",
        format_block_walkthrough(block_steps),
        "\n\n",
        get_full_image_metrics_breakdown(
            original_u8, reconstructed_u8, compressed_bytes, quality, is_color
        ),
    ]
    return "".join(sections)
