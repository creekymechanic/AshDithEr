"""Dithering algorithms and a unified entry-point for application.

Exported API
------------
- apply_dither(image_array, num_colors, method="floyd")

Supported methods
-----------------
- "none"   : no dithering; uniform per-channel quantization to a cubic grid
- "floyd"  : Floyd–Steinberg error diffusion
- "atkinson": Atkinson error diffusion
- "burkes" : Burkes error diffusion
- "sierra" : Sierra-3 error diffusion
- "bayer2" : Ordered/Bayer dithering using a 2x2 matrix
- "bayer4" : Ordered/Bayer dithering using a 4x4 matrix
- "bayer8" : Ordered/Bayer dithering using a 8x8 matrix

Implementation notes
--------------------
All dithers operate on NumPy arrays. Quantization uses a per-channel
uniform grid with L levels per channel where L^3 approximates the
requested total number of colors.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from . import floyd, atkinson, burkes, sierra, bayer

Array = np.ndarray


def _levels_from_num_colors(num_colors: int) -> int:
    """Compute per-channel levels L such that L^3 ~= num_colors (L>=2).

    This provides a uniform RGB grid with up to L^3 colors. We choose L so that
    L^3 <= num_colors, but if num_colors < 8 we still enforce L>=2 (giving 8 colors).
    """
    if num_colors < 2:
        raise ValueError("num_colors must be >= 2")
    # Start near the cubic root, adjust to fit under the limit
    L = int(round(num_colors ** (1 / 3)))
    L = max(2, L)
    while L ** 3 > num_colors and L > 2:
        L -= 1
    while (L + 1) ** 3 <= num_colors:
        L += 1
    return max(2, L)


def _quantize_none(arr: Array, L: int) -> Array:
    """Uniform per-channel quantization to L levels (no dithering)."""
    arrf = arr.astype(np.float32)
    if L <= 1:
        return np.zeros_like(arr, dtype=np.uint8)
    q = np.rint((arrf * (L - 1) / 255.0)).clip(0, L - 1)
    out = np.rint(q * (255.0 / (L - 1))).clip(0, 255).astype(np.uint8)
    return out


def apply_dither(
    image_array: Array,
    num_colors: int,
    method: Literal[
        "none", "floyd", "atkinson", "burkes", "sierra", "bayer2", "bayer4", "bayer8"
    ] = "floyd",
) -> Array:
    """Apply the selected dithering method to an image array.

    Parameters
    ----------
    image_array : np.ndarray
        RGB image array of shape (H, W, 3), dtype=uint8.
    num_colors : int
        Approximate total palette size (colors) to quantize to.
    method : str
        Dithering method to apply.

    Returns
    -------
    np.ndarray
        Dithered and quantized image array, dtype=uint8.
    """
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image_array must be an RGB array with shape (H, W, 3)")
    if num_colors < 2:
        raise ValueError("num_colors must be >= 2")

    L = _levels_from_num_colors(num_colors)

    m = method.lower()
    if m == "none":
        return _quantize_none(image_array, L)
    if m == "floyd":
        return floyd.dither_floyd(image_array, L)
    if m == "atkinson":
        return atkinson.dither_atkinson(image_array, L)
    if m == "burkes":
        return burkes.dither_burkes(image_array, L)
    if m == "sierra":
        return sierra.dither_sierra(image_array, L)
    if m == "bayer2":
        return bayer.dither_bayer(image_array, L, size=2)
    if m == "bayer4":
        return bayer.dither_bayer(image_array, L, size=4)
    if m == "bayer8":
        return bayer.dither_bayer(image_array, L, size=8)

    raise ValueError(f"Unknown dithering method: {method}")


__all__ = ["apply_dither"]
