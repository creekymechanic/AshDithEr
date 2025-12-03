"""Nearest-neighbor upscaling for NumPy arrays."""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def upscale_nearest(arr: Array, factor: int) -> Array:
    """Upscale an RGB image array by an integer factor using nearest-neighbor.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (H, W, 3), dtype=uint8.
    factor : int
        Upscale factor (>=1).

    Returns
    -------
    np.ndarray
        Upscaled image array.
    """
    if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("arr must be an RGB image with shape (H, W, 3)")
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return arr.copy()

    up = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
    return up.astype(np.uint8)
