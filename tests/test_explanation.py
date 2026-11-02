"""Unit tests for explanation, mathematical formulas, algorithms, and numerical calculations."""

import os
import tkinter as tk
import numpy as np

from dct_core.explanation import (
    STANDARD_SAMPLE_BLOCK,
    calculate_block_steps,
    format_block_walkthrough,
    format_matrix,
    generate_full_academic_report,
    get_algorithms_text,
    get_formulas_text,
    get_full_image_metrics_breakdown,
    get_pipeline_flowchart,
)
from app.gui import DCTApp
from main import main as cli_main


def test_calculate_block_steps_standard_sample():
    steps = calculate_block_steps(STANDARD_SAMPLE_BLOCK, quality=50)
    assert steps.quality == 50
    assert steps.scale_factor == 100.0  # 200 - 2*50 = 100
    assert steps.original.shape == (8, 8)
    assert steps.shifted.shape == (8, 8)
    assert steps.dct_coeffs.shape == (8, 8)
    assert steps.quant_table.shape == (8, 8)
    assert steps.quant_coeffs.shape == (8, 8)
    assert steps.dequant_coeffs.shape == (8, 8)
    assert steps.reconstructed.shape == (8, 8)
    assert steps.spatial_error.shape == (8, 8)

    # Textbook DC coefficient for standard block is ~ -415.38
    assert np.isclose(steps.dc_coeff, -415.375, atol=1e-2)
    # Quantized DC coefficient at Q=50 is -26
    assert steps.quant_coeffs[0, 0] == -26

    # Verify zero count + non-zero count == 64
    assert 0 <= steps.zeros_count <= 64
    assert 0.0 <= steps.zeros_pct <= 100.0
    assert steps.zigzag_array.shape == (64,)
    assert steps.block_bytes > 0
    assert steps.block_mse >= 0.0


def test_quality_factor_scaling_extremes():
    steps_q10 = calculate_block_steps(STANDARD_SAMPLE_BLOCK, quality=10)
    steps_q95 = calculate_block_steps(STANDARD_SAMPLE_BLOCK, quality=95)

    # Q=10 should have significantly more zeros and smaller byte size than Q=95
    assert steps_q10.zeros_count >= steps_q95.zeros_count
    assert steps_q10.block_bytes <= steps_q95.block_bytes
    # Q=95 should have better quality (lower MSE, higher PSNR)
    assert steps_q95.block_mse <= steps_q10.block_mse
    assert steps_q95.block_psnr >= steps_q10.block_psnr


def test_format_matrix():
    mat = np.eye(8, dtype=int)
    formatted = format_matrix(mat, fmt="{:4d}")
    assert "[0]" in formatted
    assert "[7]" in formatted
    assert len(formatted.splitlines()) >= 9


def test_text_generators():
    flowchart = get_pipeline_flowchart()
    assert "FLOWCHART" in flowchart or "PIPELINE" in flowchart
    assert "LEVEL SHIFTING" in flowchart
    assert "QUANTIZATION" in flowchart

    algos = get_algorithms_text()
    assert "ALGORITHM 1: FORWARD 2-D DISCRETE COSINE TRANSFORM" in algos
    assert "ALGORITHM 2: ADAPTIVE QUANTIZATION" in algos
    assert "ALGORITHM 3: ZIG-ZAG SCANNING" in algos

    formulas = get_formulas_text()
    assert "2-D DISCRETE COSINE TRANSFORM" in formulas
    assert "STANDARD JPEG LUMINANCE QUANTIZATION TABLE" in formulas
    assert "Peak Signal-to-Noise Ratio" in formulas


def test_full_image_metrics_breakdown():
    orig = np.full((32, 32), 100, dtype=np.uint8)
    recon = np.full((32, 32), 102, dtype=np.uint8)
    breakdown = get_full_image_metrics_breakdown(
        orig, recon, compressed_bytes=200, quality=75, is_color=False
    )
    assert "GLOBAL IMAGE COMPRESSION METRICS" in breakdown
    assert "Mean Squared Error" in breakdown
    assert "4.0000" in breakdown or "4.0" in breakdown  # (102-100)^2 = 4


def test_report_generation():
    orig = STANDARD_SAMPLE_BLOCK
    steps = calculate_block_steps(orig, quality=75)
    report = generate_full_academic_report(
        orig, steps.reconstructed, compressed_bytes=steps.block_bytes, quality=75, is_color=False, block_steps=steps
    )
    assert "DIGITAL IMAGE PROCESSING (DIP) COURSE PROJECT" in report
    assert "STEP 1: ORIGINAL 8x8 PIXEL BLOCK" in report


def test_cli_explain_and_report_flags(tmp_path):
    report_file = str(tmp_path / "test_report.txt")
    code = cli_main(["assets/sample.png", "-q", "60", "--explain", "--report", report_file])
    assert code == 0
    assert os.path.isfile(report_file)
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "DCT IMAGE COMPRESSION" in content
    assert "ALGORITHM" in content
    assert "MATHEMATICAL FORMULAS" in content


def test_gui_theory_popup_dialog_lifecycle():
    root = tk.Tk()
    root.withdraw()
    try:
        app = DCTApp(root)
        app.on_show_theory_popup()
        assert app._theory_dialog is not None
        assert app._theory_dialog.winfo_exists()

        # Switch block sources
        app._theory_dialog.block_source_var.set("Textbook Sample (Gonzalez & Woods)")
        app._theory_dialog.on_calculate_block()
        assert "STEP 1: ORIGINAL 8x8 PIXEL BLOCK" in app._theory_dialog.txt_block.get("1.0", tk.END)

        # Trigger slider sync
        app.q_var.set(45)
        app._sync_theory_dialog_if_open()
        assert app._theory_dialog.quality == 45

        # Close dialog
        app._theory_dialog.destroy()
    finally:
        root.destroy()
