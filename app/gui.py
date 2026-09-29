"""
app.gui
=======
Tkinter desktop app for the DCT compression demo.

Layout:
    +---------------------------------------------------------+
    |  [Open Image...]  Path: <shown>                         |
    |  +--------------------+   +---------------------------+ |
    |  |                    |   |                           | |
    |  |   Original         |   |   Compressed @ Q = 75     | |
    |  |                    |   |                           | |
    |  +--------------------+   +---------------------------+ |
    |  Quality: [====|===========]  Q=75                      |
    |  Size: 512 KB → 184 KB   CR: 2.78x                     |
    |  MSE: 17.42   PSNR: 35.7 dB                             |
    |  [Save reconstruction]  [Save 4-up comparison grid]     |
    +---------------------------------------------------------+

The original image is shown once. The right-hand preview is recomputed
on every slider movement (with a small debounce so dragging is smooth).
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from io_utils.io import load_image, make_comparison_grid, save_image
from pipeline import compress_image


def _to_tk_image(arr: np.ndarray, size: int = 360) -> ImageTk.PhotoImage:
    """Convert numpy uint8 array → Tk PhotoImage at the given square size."""
    pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize((size, size), Image.LANCZOS)
    return ImageTk.PhotoImage(pil)


class DCTApp:
    """Top-level Tk controller."""

    PREVIEW_SIZE = 360
    DEBOUNCE_MS = 120

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DCT Image Compression — Quality Analysis")
        self.root.configure(bg="#f5f5f7")

        self.image_path: str | None = None
        self.image_u8: np.ndarray | None = None
        self.is_color: bool = False

        self._tk_orig: ImageTk.PhotoImage | None = None
        self._tk_comp: ImageTk.PhotoImage | None = None
        self._pending_after: str | None = None

        self._build_widgets()

    # ---------- layout ----------------------------------------------------
    def _build_widgets(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#f5f5f7")
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Metric.TLabel", font=("Consolas", 10))
        style.configure("TButton", padding=6)

        # Top bar: open + path
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Open Image...", command=self.on_open).pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value="(no image loaded)")
        ttk.Label(top, textvariable=self.path_var, padding=(10, 0)).pack(side=tk.LEFT)

        # Preview row
        preview_row = ttk.Frame(self.root, padding=(10, 0))
        preview_row.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        orig_frame = ttk.Frame(preview_row)
        comp_frame = ttk.Frame(preview_row)
        orig_frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True)
        comp_frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True)
        ttk.Label(orig_frame, text="Original", style="Header.TLabel").pack()
        ttk.Label(comp_frame, text="Compressed", style="Header.TLabel").pack()
        self.orig_canvas = tk.Label(orig_frame, bg="#222")
        self.orig_canvas.pack(pady=6)
        self.comp_canvas = tk.Label(comp_frame, bg="#222")
        self.comp_canvas.pack(pady=6)

        # Quality slider
        slider_row = ttk.Frame(self.root, padding=(10, 6))
        slider_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(slider_row, text="Quality:").pack(side=tk.LEFT)
        self.q_var = tk.IntVar(value=75)
        self.slider = ttk.Scale(slider_row, from_=1, to=100, orient=tk.HORIZONTAL,
                                variable=self.q_var, command=self.on_slider_change)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.q_readout = ttk.Label(slider_row, text="Q = 75", width=8)
        self.q_readout.pack(side=tk.LEFT)

        # Metrics panel
        metrics = ttk.Frame(self.root, padding=(10, 6))
        metrics.pack(side=tk.TOP, fill=tk.X)
        self.metrics_var = tk.StringVar(value="Load an image to begin.")
        ttk.Label(metrics, textvariable=self.metrics_var, style="Metric.TLabel").pack(anchor=tk.W)

        # Action buttons
        actions = ttk.Frame(self.root, padding=10)
        actions.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(actions, text="Save Reconstruction",
                   command=self.on_save_recon).pack(side=tk.LEFT)
        ttk.Button(actions, text="Save 4-up Comparison Grid",
                   command=self.on_save_grid).pack(side=tk.LEFT, padx=8)
                   
        ttk.Label(actions, text="Format:").pack(side=tk.LEFT, padx=(20, 4))
        self.format_var = tk.StringVar(value="PNG")
        self.format_dropdown = ttk.Combobox(actions, textvariable=self.format_var,
                                            values=["PNG", "JPG"], width=5, state="readonly")
        self.format_dropdown.pack(side=tk.LEFT)

        ttk.Button(actions, text="Quit", command=self.root.destroy).pack(side=tk.RIGHT)

    # ---------- event handlers -------------------------------------------
    def on_open(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        try:
            arr, is_color = load_image(path)
        except Exception as exc:  # surface a friendly error if format is wrong
            messagebox.showerror("Could not load image", str(exc))
            return
        self.image_path = path
        self.image_u8 = arr
        self.is_color = is_color
        self.path_var.set(os.path.basename(path))
        self._tk_orig = _to_tk_image(arr, size=self.PREVIEW_SIZE)
        self.orig_canvas.configure(image=self._tk_orig)
        self._schedule_update(immediate=True)

    def on_slider_change(self, _evt=None) -> None:
        self.q_readout.configure(text=f"Q = {self.q_var.get()}")
        self._schedule_update()

    def _schedule_update(self, immediate: bool = False) -> None:
        if self._pending_after is not None:
            try:
                self.root.after_cancel(self._pending_after)
            except tk.TclError:
                pass
            self._pending_after = None
        delay = 0 if immediate else self.DEBOUNCE_MS
        self._pending_after = self.root.after(delay, self._refresh_compressed)

    def _refresh_compressed(self) -> None:
        self._pending_after = None
        if self.image_u8 is None:
            return
        q = int(self.q_var.get())
        try:
            result = compress_image(self.image_u8, quality=q)
        except Exception as exc:
            messagebox.showerror("Compression failed", str(exc))
            return
        self._last_result = result
        self._tk_comp = _to_tk_image(result.reconstructed, size=self.PREVIEW_SIZE)
        self.comp_canvas.configure(image=self._tk_comp)

        psnr_str = "inf" if result.psnr_val == float("inf") else f"{result.psnr_val:.2f} dB"
        text = (
            f"Q = {q}   |   "
            f"Original: {result.original_bytes / 1024:.1f} KB   "
            f"Compressed: {result.compressed_bytes / 1024:.1f} KB   "
            f"CR = {result.cr:.2f} x\n"
            f"MSE = {result.mse_val:.3f}   PSNR = {psnr_str}   "
            f"{'colour' if self.is_color else 'grayscale'}"
        )
        self.metrics_var.set(text)

    def on_save_recon(self) -> None:
        if not hasattr(self, "_last_result"):
            messagebox.showinfo("Nothing to save", "Open and compress an image first.")
            return
        
        ext = self.format_var.get().lower()
        default = f"reconstruction.{ext}"
        if self.image_path:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            default = f"{base}_recon_q{self.q_var.get()}.{ext}"
            
        filetypes = [("PNG", "*.png")] if ext == "png" else [("JPEG", "*.jpg *.jpeg")]
        out = filedialog.asksaveasfilename(defaultextension=f".{ext}",
                                           initialfile=default,
                                           filetypes=filetypes)
        if not out:
            return
        path = save_image(self._last_result.reconstructed, out)
        messagebox.showinfo("Saved", f"Saved to:\n{path}")

    def on_save_grid(self) -> None:
        if self.image_u8 is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        levels = [90, 75, 50, 10]
        images = [self.image_u8]
        labels = ["Original"]
        for q in levels[1:]:
            res = compress_image(self.image_u8, quality=q)
            images.append(res.reconstructed)
            labels.append(f"Q = {q}")
            
        ext = self.format_var.get().lower()
        default = f"comparison_grid.{ext}"
        if self.image_path:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            default = f"{base}_comparison.{ext}"
            
        filetypes = [("PNG", "*.png")] if ext == "png" else [("JPEG", "*.jpg *.jpeg")]
        out = filedialog.asksaveasfilename(defaultextension=f".{ext}",
                                           initialfile=default,
                                           filetypes=filetypes)
        if not out:
            return
        path = make_comparison_grid(images, labels, out, tile_size=256)
        messagebox.showinfo("Saved", f"Saved to:\n{path}")


def main() -> int:
    root = tk.Tk()
    DCTApp(root)
    root.geometry("900x720")
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
