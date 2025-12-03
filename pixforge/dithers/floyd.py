"""Floyd–Steinberg error diffusion dithering.

This module uses an optional Numba-accelerated implementation for speed.
If Numba is unavailable, it falls back to a pure-NumPy+Python loop.
"""
from __future__ import annotations

import numpy as np

try:  # Optional acceleration
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None  # type: ignore

Array = np.ndarray


def _quantize_levels(pixel: np.ndarray, L: int) -> np.ndarray:
    """Quantize a single RGB pixel to L uniform levels per channel."""
    q = np.rint(pixel * (L - 1) / 255.0)
    q = np.clip(q, 0, L - 1)
    out = np.rint(q * (255.0 / (L - 1)))
    return np.clip(out, 0, 255)


def _has_numba() -> bool:
    return njit is not None


if njit is not None:  # pragma: no cover - requires numba at runtime
    @njit(cache=True)
    def _floyd_impl(work: np.ndarray, L: int, serpentine: bool) -> None:
        H, W, _ = work.shape
        step = 255.0 / (L - 1)
        for y in range(H):
            if serpentine and (y % 2 == 1):
                start, stop, step_dir = W - 1, -1, -1
            else:
                start, stop, step_dir = 0, W, 1
            x = start
            while x != stop:
                old0 = work[y, x, 0]
                old1 = work[y, x, 1]
                old2 = work[y, x, 2]
                # Quantize each channel
                q0 = round(old0 * (L - 1) / 255.0)
                q1 = round(old1 * (L - 1) / 255.0)
                q2 = round(old2 * (L - 1) / 255.0)
                if q0 < 0:
                    q0 = 0
                elif q0 > (L - 1):
                    q0 = L - 1
                if q1 < 0:
                    q1 = 0
                elif q1 > (L - 1):
                    q1 = L - 1
                if q2 < 0:
                    q2 = 0
                elif q2 > (L - 1):
                    q2 = L - 1
                new0 = round(q0 * step)
                new1 = round(q1 * step)
                new2 = round(q2 * step)
                work[y, x, 0] = new0
                work[y, x, 1] = new1
                work[y, x, 2] = new2
                err0 = old0 - new0
                err1 = old1 - new1
                err2 = old2 - new2

                xn1 = x + step_dir
                if 0 <= xn1 < W:
                    work[y, xn1, 0] += err0 * (7.0 / 16.0)
                    work[y, xn1, 1] += err1 * (7.0 / 16.0)
                    work[y, xn1, 2] += err2 * (7.0 / 16.0)
                if y + 1 < H:
                    xp1 = x - step_dir
                    if 0 <= xp1 < W:
                        work[y + 1, xp1, 0] += err0 * (3.0 / 16.0)
                        work[y + 1, xp1, 1] += err1 * (3.0 / 16.0)
                        work[y + 1, xp1, 2] += err2 * (3.0 / 16.0)
                    work[y + 1, x, 0] += err0 * (5.0 / 16.0)
                    work[y + 1, x, 1] += err1 * (5.0 / 16.0)
                    work[y + 1, x, 2] += err2 * (5.0 / 16.0)
                    if 0 <= xn1 < W:
                        work[y + 1, xn1, 0] += err0 * (1.0 / 16.0)
                        work[y + 1, xn1, 1] += err1 * (1.0 / 16.0)
                        work[y + 1, xn1, 2] += err2 * (1.0 / 16.0)
                x += step_dir



def dither_floyd(arr: Array, L: int, serpentine: bool = True) -> Array:
    """Apply Floyd–Steinberg dithering with uniform RGB grid quantization.

    Parameters
    ----------
    arr : np.ndarray
        Input RGB image (H, W, 3), dtype=uint8.
    L : int
        Levels per channel. The output palette has up to L^3 colors.
    serpentine : bool
        If True, alternate scan direction by row to reduce artifacts.

    Returns
    -------
    np.ndarray
        Dithered image (uint8).
    """
    H, W, _ = arr.shape
    work = arr.astype(np.float32).copy()

    if _has_numba():  # Use accelerated path
        _floyd_impl(work, L, serpentine)  # type: ignore[arg-type]
    else:
        for y in range(H):
            # Determine scan direction
            if serpentine and (y % 2 == 1):
                x_range = range(W - 1, -1, -1)
                dir_sign = -1
            else:
                x_range = range(W)
                dir_sign = 1

            for x in x_range:
                old = work[y, x]
                new = _quantize_levels(old, L)
                work[y, x] = new
                err = old - new

                # Floyd–Steinberg kernel (normalized by 16):
                #   *   7
                #  3  5  1
                # Adjust direction for serpentine scanning
                nx = x + dir_sign
                if 0 <= nx < W:
                    work[y, nx] += err * (7 / 16.0)
                if y + 1 < H:
                    px = x - dir_sign
                    if 0 <= px < W:
                        work[y + 1, px] += err * (3 / 16.0)
                    work[y + 1, x] += err * (5 / 16.0)
                    if 0 <= nx < W:
                        work[y + 1, nx] += err * (1 / 16.0)

    return np.clip(np.rint(work), 0, 255).astype(np.uint8)
