"""Tests for io_utils."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from io_utils.io import load_image, make_comparison_grid, save_image


def test_roundtrip_png_grayscale(tmp_path=None):
    tmp = tempfile.mkdtemp()
    img = (np.random.default_rng(5).random((48, 48)) * 255).astype(np.uint8)
    path = os.path.join(tmp, "g.png")
    saved = save_image(img, path)
    loaded, is_color = load_image(saved)
    assert is_color is False
    np.testing.assert_array_equal(loaded, img)


def test_roundtrip_png_color(tmp_path=None):
    tmp = tempfile.mkdtemp()
    img = (np.random.default_rng(6).integers(0, 256, (40, 60, 3), dtype=np.uint8))
    path = os.path.join(tmp, "c.png")
    saved = save_image(img, path)
    loaded, is_color = load_image(saved)
    assert is_color is True
    np.testing.assert_array_equal(loaded, img)


def test_comparison_grid_writes_file(tmp_path=None):
    tmp = tempfile.mkdtemp()
    a = np.full((20, 20, 3), 10, dtype=np.uint8)
    b = np.full((20, 20, 3), 200, dtype=np.uint8)
    out = os.path.join(tmp, "grid.png")
    path = make_comparison_grid([a, b], ["L", "H"], out, tile_size=40)
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0
