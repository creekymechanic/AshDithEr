"""Atkinson error diffusion dithering.

This module uses an optional Numba-accelerated implementation for speed.
Falls back to a pure-NumPy+Python loop if Numba isn't available.
"""
from __future__ import annotations

import numpy as np

try:  # Optional acceleration
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None  # type: ignore

Array = np.ndarray


def _quantize_levels(pixel: np.ndarray, L: int) -> np.ndarray:
    q = np.rint(pixel * (L - 1) / 255.0)
    q = np.clip(q, 0, L - 1)
    out = np.rint(q * (255.0 / (L - 1)))
    return np.clip(out, 0, 255)


def dither_atkinson(arr: Array, L: int, serpentine: bool = True) -> Array:
    """Apply Atkinson dithering with uniform RGB grid quantization.

    The Atkinson kernel diffuses error to 6 neighbors with weight 1/8 each;
    the total distributed weight is 6/8, allowing some error to dissipate,
    which tends to produce a lighter result.
    """
    H, W, _ = arr.shape
    work = arr.astype(np.float32).copy()

    if njit is not None:  # pragma: no cover
        @njit(cache=True)
        def _atk_impl(work: np.ndarray, L: int, serpentine: bool) -> None:
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
                        work[y, xn1, 0] += err0 * 0.125
                        work[y, xn1, 1] += err1 * 0.125
                        work[y, xn1, 2] += err2 * 0.125
                    xn2 = x + 2 * step_dir
                    if 0 <= xn2 < W:
                        work[y, xn2, 0] += err0 * 0.125
                        work[y, xn2, 1] += err1 * 0.125
                        work[y, xn2, 2] += err2 * 0.125
                    if y + 1 < H:
                        xp1 = x - step_dir
                        if 0 <= xp1 < W:
                            work[y + 1, xp1, 0] += err0 * 0.125
                            work[y + 1, xp1, 1] += err1 * 0.125
                            work[y + 1, xp1, 2] += err2 * 0.125
                        work[y + 1, x, 0] += err0 * 0.125
                        work[y + 1, x, 1] += err1 * 0.125
                        work[y + 1, x, 2] += err2 * 0.125
                        if 0 <= xn1 < W:
                            work[y + 1, xn1, 0] += err0 * 0.125
                            work[y + 1, xn1, 1] += err1 * 0.125
                            work[y + 1, xn1, 2] += err2 * 0.125
                    if y + 2 < H:
                        work[y + 2, x, 0] += err0 * 0.125
                        work[y + 2, x, 1] += err1 * 0.125
                        work[y + 2, x, 2] += err2 * 0.125
                    x += step_dir

        _atk_impl(work, L, serpentine)
    else:
        for y in range(H):
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

                if 0 <= x + dir_sign < W:
                    work[y, x + dir_sign] += err * 0.125
                if 0 <= x + 2 * dir_sign < W:
                    work[y, x + 2 * dir_sign] += err * 0.125
                if y + 1 < H:
                    if 0 <= x - dir_sign < W:
                        work[y + 1, x - dir_sign] += err * 0.125
                    work[y + 1, x] += err * 0.125
                    if 0 <= x + dir_sign < W:
                        work[y + 1, x + dir_sign] += err * 0.125
                if y + 2 < H:
                    work[y + 2, x] += err * 0.125

    return np.clip(np.rint(work), 0, 255).astype(np.uint8)
