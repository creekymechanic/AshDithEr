"""Image loading and saving utilities using Pillow, with NumPy arrays.

All processing in this project should occur on NumPy arrays. These helpers
only convert between Pillow images and NumPy `uint8` RGB arrays for IO.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


Array = np.ndarray


def load_image(path: Union[str, Path]) -> Array:
    """Load an image file into an RGB NumPy array (uint8).

    Parameters
    ----------
    path : str | Path
        Path to an image supported by Pillow.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, 3), dtype=uint8, in RGB order.
    """
    p = Path(path)
    with Image.open(p) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.uint8)
    return arr


def save_image(arr: Array, path: Union[str, Path]) -> None:
    """Save an RGB NumPy array (uint8) to an image file via Pillow.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (H, W, 3), dtype=uint8.
    path : str | Path
        Output file path. The format is inferred from the extension.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a NumPy array")
    if arr.dtype != np.uint8:
        raise TypeError("arr must have dtype=uint8")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("arr must have shape (H, W, 3)")

    p = Path(path)
    im = Image.fromarray(arr, mode="RGB")
    im.save(p)
