"""Nearest-neighbor resizing utilities for NumPy arrays.

Provides integer-agnostic nearest-neighbor scaling to arbitrary output size
for crisp pixel-art operations.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def resize_nearest(arr: Array, new_h: int, new_w: int) -> Array:
    """Resize an RGB image to (new_h, new_w) via nearest-neighbor.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (H, W, 3), dtype=uint8.
    new_h : int
        Target height (>=1).
    new_w : int
        Target width (>=1).

    Returns
    -------
    np.ndarray
        Resized image.
    """
    if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("arr must be an RGB image with shape (H, W, 3)")
    if new_h < 1 or new_w < 1:
        raise ValueError("new_h and new_w must be >= 1")

    H, W, C = arr.shape
    if H == new_h and W == new_w:
        return arr.copy()

    # Map output pixel centers to input indices
    y = (np.arange(new_h) * (H / new_h)).astype(np.float32)
    x = (np.arange(new_w) * (W / new_w)).astype(np.float32)
    yi = np.clip(np.rint(y), 0, H - 1).astype(np.int64)
    xi = np.clip(np.rint(x), 0, W - 1).astype(np.int64)

    out = arr[yi[:, None], xi[None, :], :]
    return out.astype(np.uint8)


def resize_nearest_scale(arr: Array, scale: float) -> Array:
    """Resize an RGB image by a float ``scale`` via nearest-neighbor.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (H, W, 3), dtype=uint8.
    scale : float
        Scale factor (>0). Values >1 upscale, <1 downscale.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")
    H, W, _ = arr.shape
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    return resize_nearest(arr, new_h, new_w)
