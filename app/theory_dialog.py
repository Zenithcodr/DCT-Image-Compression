"""
app.theory_dialog
=================
Toplevel educational pop-up window for Digital Image Processing (DIP) course evaluation.

Provides:
  - Tab 1: 📋 Complete Pipeline Flowchart
  - Tab 2: ⚙️ Formal DIP Algorithms & Pseudocode
  - Tab 3: 📐 Mathematical Formulas Reference
  - Tab 4: 🔢 Live 8x8 Block Numerical Calculation Walkthrough
  - Tab 5: 📊 Global Image Metrics Breakdown & Verification
  - Export Academic Report to file (.txt)
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Optional

import numpy as np

from dct_core.blocks import split_blocks
from dct_core.explanation import (
    STANDARD_SAMPLE_BLOCK,
    BlockCalculationSteps,
    calculate_block_steps,
    format_block_walkthrough,
    generate_full_academic_report,
    get_algorithms_text,
    get_formulas_text,
    get_full_image_metrics_breakdown,
    get_pipeline_flowchart,
)
from pipeline.color import rgb_to_ycbcr


class TheoryCalculationDialog(tk.Toplevel):
    """Educational pop-up modal/window showing theory, algorithms, formulas,
    and live numerical calculations."""

    def __init__(
        self,
        parent: tk.Tk,
        image_u8: Optional[np.ndarray],
        quality: int,
        is_color: bool,
        last_result: Any = None,
    ):
        super().__init__(parent)
        self.title("DIP Theory, Flowchart, Algorithms & Numerical Calculations")
        self.geometry("1020x760")
        self.minsize(820, 600)
        self.configure(bg="#f1f5f9")

        self.parent = parent
        self.image_u8 = image_u8
        self.quality = quality
        self.is_color = is_color
        self.last_result = last_result

        # Internal block extraction cache
        self.current_block: np.ndarray = STANDARD_SAMPLE_BLOCK
        self.block_source_name = "Textbook Sample (Gonzalez & Woods)"
        self.current_channel_name = "Y (Luminance)"

        self._build_ui()
        self._update_block_selection()
        self._refresh_all_tabs()

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Header banner
        header = tk.Frame(self, bg="#1e293b", padx=16, pady=12)
        header.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="Digital Image Processing: DCT Image Compression Architecture",
            font=("Segoe UI", 13, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = tk.Label(
            header,
            text="Interactive viva & presentation tool: Complete Flowchart, Algorithms, Governing Equations & Live Numerical Block Calculations",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b",
        )
        subtitle_lbl.pack(anchor=tk.W, pady=(2, 0))

        # Bottom action bar
        bottom_bar = tk.Frame(self, bg="#e2e8f0", padx=14, pady=10)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(
            bottom_bar,
            text="💾 Export Complete Academic Report (.txt)",
            command=self.on_export_report,
        ).pack(side=tk.LEFT)

        ttk.Button(
            bottom_bar,
            text="🔄 Refresh From Main GUI",
            command=self.on_refresh_from_parent,
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            bottom_bar,
            text="Close",
            command=self.destroy,
        ).pack(side=tk.RIGHT)

        # Tabbed notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Flowchart
        self.tab_flowchart = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_flowchart, text="📋 1. Pipeline Flowchart")
        self.txt_flowchart = self._create_scrolled_text(self.tab_flowchart)

        # Tab 2: Algorithms
        self.tab_algos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_algos, text="⚙️ 2. Step-by-Step Algorithms")
        self.txt_algos = self._create_scrolled_text(self.tab_algos)

        # Tab 3: Formulas
        self.tab_formulas = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_formulas, text="📐 3. Mathematical Formulas")
        self.txt_formulas = self._create_scrolled_text(self.tab_formulas)

        # Tab 4: Live 8x8 Block Walkthrough
        self.tab_block = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_block, text="🔢 4. Live 8x8 Numerical Walkthrough")
        self._build_block_tab()

        # Tab 5: Image Metrics Breakdown
        self.tab_metrics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metrics, text="📊 5. Image Metrics Derivation")
        self.txt_metrics = self._create_scrolled_text(self.tab_metrics)

    def _create_scrolled_text(self, parent: tk.Widget) -> ScrolledText:
        st = ScrolledText(
            parent,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#0f172a",
            insertbackground="#0f172a",
            padx=12,
            pady=12,
            relief=tk.FLAT,
        )
        st.pack(fill=tk.BOTH, expand=True)
        return st

    def _build_block_tab(self) -> None:
        # Control strip at top of Tab 4
        ctrl = tk.Frame(self.tab_block, bg="#f8fafc", padx=10, pady=8, bd=1, relief=tk.SOLID)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        tk.Label(ctrl, text="Block Source:", font=("Segoe UI", 9, "bold"), bg="#f8fafc").pack(side=tk.LEFT)
        self.block_source_var = tk.StringVar(value="Textbook Sample (Gonzalez & Woods)")
        self.block_source_combo = ttk.Combobox(
            ctrl,
            textvariable=self.block_source_var,
            state="readonly",
            width=36,
        )
        self.block_source_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.block_source_combo.bind("<<ComboboxSelected>>", self.on_block_source_change)

        # Channel selector (for color images)
        self.lbl_channel = tk.Label(ctrl, text="Channel:", font=("Segoe UI", 9, "bold"), bg="#f8fafc")
        self.channel_var = tk.StringVar(value="Y (Luminance)")
        self.channel_combo = ttk.Combobox(
            ctrl,
            textvariable=self.channel_var,
            values=["Y (Luminance)", "Cb (Chrominance)", "Cr (Chrominance)"],
            state="readonly",
            width=18,
        )
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_change)

        # Quality factor spinner
        tk.Label(ctrl, text="Quality (Q):", font=("Segoe UI", 9, "bold"), bg="#f8fafc").pack(side=tk.LEFT, padx=(4, 4))
        self.q_var = tk.IntVar(value=self.quality)
        self.q_spin = ttk.Spinbox(
            ctrl,
            from_=1,
            to=100,
            textvariable=self.q_var,
            width=5,
            command=self.on_q_spin_change,
        )
        self.q_spin.pack(side=tk.LEFT)
        self.q_spin.bind("<Return>", lambda e: self.on_q_spin_change())

        ttk.Button(ctrl, text="Calculate Block", command=self.on_calculate_block).pack(side=tk.LEFT, padx=12)

        self.txt_block = self._create_scrolled_text(self.tab_block)

    # -------------------------------------------------------------------------
    # Data & Tab Population
    # -------------------------------------------------------------------------
    def _update_block_selection(self) -> None:
        """Update available block options based on whether an image is loaded."""
        options = ["Textbook Sample (Gonzalez & Woods)"]

        if self.image_u8 is not None:
            options.append("Loaded Image: Top-Left Block (0, 0)")
            options.append("Loaded Image: Center Block")
            options.append("Loaded Image: High-Texture Block")
            if self.is_color:
                self.lbl_channel.pack(side=tk.LEFT)
                self.channel_combo.pack(side=tk.LEFT, padx=(4, 12))
            else:
                self.lbl_channel.pack_forget()
                self.channel_combo.pack_forget()
        else:
            self.lbl_channel.pack_forget()
            self.channel_combo.pack_forget()

        self.block_source_combo["values"] = options
        if self.block_source_var.get() not in options:
            self.block_source_var.set(options[0])

    def _extract_active_block(self) -> np.ndarray:
        """Extract the 8x8 block requested by the user."""
        choice = self.block_source_var.get()
        if self.image_u8 is None or choice == "Textbook Sample (Gonzalez & Woods)":
            return STANDARD_SAMPLE_BLOCK

        # Determine 2D channel
        if self.is_color and self.image_u8.ndim == 3:
            ycbcr = rgb_to_ycbcr(self.image_u8)
            ch_choice = self.channel_var.get()
            if "Cb" in ch_choice:
                channel = ycbcr[..., 1]
            elif "Cr" in ch_choice:
                channel = ycbcr[..., 2]
            else:
                channel = ycbcr[..., 0]
        else:
            channel = self.image_u8 if self.image_u8.ndim == 2 else self.image_u8[..., 0]

        blocks, (h, w), (hp, wp) = split_blocks(channel)
        n_rows = hp // 8
        n_cols = wp // 8
        total_blocks = n_rows * n_cols

        if "Top-Left" in choice or total_blocks == 1:
            idx = 0
        elif "Center" in choice:
            mid_r = n_rows // 2
            mid_c = n_cols // 2
            idx = mid_r * n_cols + mid_c
        elif "High-Texture" in choice:
            # Pick block with highest spatial standard deviation (high variance)
            stds = [np.std(blocks[i]) for i in range(min(total_blocks, 500))]
            idx = int(np.argmax(stds))
        else:
            idx = 0

        idx = max(0, min(idx, total_blocks - 1))
        return np.clip(blocks[idx], 0, 255).astype(np.uint8)

    def _refresh_all_tabs(self) -> None:
        """Populate or update content in all 5 tabs."""
        # Tab 1: Flowchart
        self._set_text(self.txt_flowchart, get_pipeline_flowchart())

        # Tab 2: Algorithms
        self._set_text(self.txt_algos, get_algorithms_text())

        # Tab 3: Formulas
        self._set_text(self.txt_formulas, get_formulas_text())

        # Tab 4: Block Walkthrough
        self.on_calculate_block()

        # Tab 5: Image Metrics
        self._refresh_metrics_tab()

    def _refresh_metrics_tab(self) -> None:
        """Update Tab 5 with global image metrics derivation."""
        if self.image_u8 is not None and self.last_result is not None:
            text = get_full_image_metrics_breakdown(
                original_u8=self.image_u8,
                reconstructed_u8=self.last_result.reconstructed,
                compressed_bytes=self.last_result.compressed_bytes,
                quality=int(self.q_var.get()),
                is_color=self.is_color,
            )
        else:
            sample_orig = np.tile(STANDARD_SAMPLE_BLOCK, (64, 64))  # 512x512 demo
            sample_steps = calculate_block_steps(STANDARD_SAMPLE_BLOCK, quality=int(self.q_var.get()))
            sample_recon = np.tile(sample_steps.reconstructed, (64, 64))
            sample_comp_bytes = sample_steps.block_bytes * 4096
            text = (
                "NOTE: No image currently open in main window.\n"
                "Showing sample derivation for a standard 512x512 Grayscale test image:\n\n"
                + get_full_image_metrics_breakdown(
                    original_u8=sample_orig,
                    reconstructed_u8=sample_recon,
                    compressed_bytes=sample_comp_bytes,
                    quality=int(self.q_var.get()),
                    is_color=False,
                )
            )
        self._set_text(self.txt_metrics, text)

    def _set_text(self, text_widget: ScrolledText, content: str) -> None:
        """Safely update text in a ScrolledText widget and keep it read-only."""
        text_widget.configure(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, content)
        text_widget.configure(state=tk.DISABLED)

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------
    def on_block_source_change(self, _evt=None) -> None:
        self.on_calculate_block()

    def on_channel_change(self, _evt=None) -> None:
        self.on_calculate_block()

    def on_q_spin_change(self) -> None:
        try:
            q = int(self.q_var.get())
            if not 1 <= q <= 100:
                return
        except ValueError:
            return
        self.on_calculate_block()
        self._refresh_metrics_tab()

    def on_calculate_block(self) -> None:
        try:
            q = int(self.q_var.get())
            q = max(1, min(100, q))
        except ValueError:
            q = 75
            self.q_var.set(75)

        block = self._extract_active_block()
        steps = calculate_block_steps(block, quality=q)
        self._current_steps = steps

        walkthrough_text = format_block_walkthrough(steps)
        self._set_text(self.txt_block, walkthrough_text)

    def on_refresh_from_parent(self) -> None:
        """Pull the latest state from the parent DCTApp window."""
        if hasattr(self.parent, "image_u8"):
            self.image_u8 = getattr(self.parent, "image_u8")
        if hasattr(self.parent, "is_color"):
            self.is_color = getattr(self.parent, "is_color")
        if hasattr(self.parent, "q_var"):
            self.quality = int(getattr(self.parent, "q_var").get())
            self.q_var.set(self.quality)
        if hasattr(self.parent, "_last_result"):
            self.last_result = getattr(self.parent, "_last_result")

        self._update_block_selection()
        self._refresh_all_tabs()
        messagebox.showinfo("Refreshed", "Theory and calculations updated from main window.")

    def on_export_report(self) -> None:
        """Export comprehensive DIP calculation report to a text or markdown file."""
        q = int(self.q_var.get())
        block = self._extract_active_block()
        steps = calculate_block_steps(block, quality=q)

        if self.image_u8 is not None and self.last_result is not None:
            orig = self.image_u8
            recon = self.last_result.reconstructed
            comp_bytes = self.last_result.compressed_bytes
            color = self.is_color
        else:
            orig = np.tile(STANDARD_SAMPLE_BLOCK, (64, 64))
            steps_std = calculate_block_steps(STANDARD_SAMPLE_BLOCK, quality=q)
            recon = np.tile(steps_std.reconstructed, (64, 64))
            comp_bytes = steps_std.block_bytes * 4096
            color = False

        report = generate_full_academic_report(
            original_u8=orig,
            reconstructed_u8=recon,
            compressed_bytes=comp_bytes,
            quality=q,
            is_color=color,
            block_steps=steps,
        )

        default_name = f"DIP_DCT_Compression_Report_Q{q}.txt"
        file_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Academic Calculation Report",
            initialfile=default_name,
            filetypes=[("Text Files", "*.txt"), ("Markdown Files", "*.md"), ("All Files", "*.*")],
            defaultextension=".txt",
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)
            messagebox.showinfo(
                "Report Saved",
                f"Full academic calculation report successfully saved to:\n{file_path}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Export Failed", f"Failed to save report:\n{exc}", parent=self)
