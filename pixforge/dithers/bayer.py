"""Ordered/Bayer dithering with variable matrix sizes (2x2, 4x4, 8x8)."""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def _bayer_matrix(n: int) -> np.ndarray:
    """Generate an n x n Bayer matrix (n must be a power of 2).

    The matrix values range from 0..n*n-1. Typical sizes used are 2, 4, 8.
    """
    if n & (n - 1) != 0 or n <= 0:
        raise ValueError("Bayer size must be a positive power of 2 (e.g., 2, 4, 8)")

    def build(k: int) -> np.ndarray:
        if k == 1:
            return np.array([[0]], dtype=np.int32)
        prev = build(k // 2)
        a = 4 * prev
        return np.block(
            [
                [a + 0, a + 2],
                [a + 3, a + 1],
            ]
        )

    return build(n)


def _quantize_none(arr: Array, L: int) -> Array:
    arrf = arr.astype(np.float32)
    if L <= 1:
        return np.zeros_like(arr, dtype=np.uint8)
    q = np.rint((arrf * (L - 1) / 255.0)).clip(0, L - 1)
    out = np.rint(q * (255.0 / (L - 1))).clip(0, 255).astype(np.uint8)
    return out


def dither_bayer(arr: Array, L: int, size: int = 4) -> Array:
    """Apply ordered dithering using an `size x size` Bayer matrix.

    Parameters
    ----------
    arr : np.ndarray
        Input RGB image (H, W, 3), dtype=uint8.
    L : int
        Levels per channel (>=2). Output palette has up to L^3 colors.
    size : int
        Bayer matrix size (power of 2): 2, 4, 8.

    Returns
    -------
    np.ndarray
        Dithered image (uint8).
    """
    if L < 2:
        return _quantize_none(arr, 2)

    H, W, _ = arr.shape
    M = _bayer_matrix(size)
    n2 = float(size * size)
    # Normalize to [0,1) then center around 0 with -0.5..+0.5
    T = (M.astype(np.float32) / n2) - 0.5

    # Prepare tiled threshold map matching the image size
    ty = (H + size - 1) // size
    tx = (W + size - 1) // size
    thresh = np.tile(T, (ty, tx))[:H, :W]

    arrf = arr.astype(np.float32)

    # Per-channel ordered quantization
    out = np.empty_like(arr, dtype=np.uint8)
    if L == 1:
        return np.zeros_like(arr, dtype=np.uint8)

    for c in range(3):
        v = arrf[:, :, c]
        # Add bias scaled relative to level step. Level step in [0..255] is 255/(L-1).
        step = 255.0 / (L - 1)
        v_bias = v + thresh * step
        q = np.floor((v_bias / 255.0) * L).clip(0, L - 1)
        out[:, :, c] = np.rint(q * step).clip(0, 255).astype(np.uint8)

    return out
