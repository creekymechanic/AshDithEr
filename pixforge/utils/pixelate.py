"""Pixelation utilities operating on NumPy arrays.

Pixelation is implemented by block-averaging the image into tiles of size
`factor x factor`, then upscaling these blocks back using nearest-neighbor.
This results in a classic pixelated effect with square blocks.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def _pad_to_multiple(arr: Array, factor: int) -> tuple[Array, tuple[int, int]]:
    """Pad array along H and W so both are multiples of `factor`.

    Pads using edge values so the visual effect remains consistent at borders.

    Returns the padded array and the original (H, W) for later cropping.
    """
    h, w, c = arr.shape
    pad_h = (factor - (h % factor)) % factor
    pad_w = (factor - (w % factor)) % factor
    if pad_h == 0 and pad_w == 0:
        return arr, (h, w)

    pad_top, pad_left = 0, 0
    pad_bottom, pad_right = pad_h, pad_w
    padded = np.pad(
        arr,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )
    return padded, (h, w)


def pixelate(arr: Array, factor: int) -> Array:
    """Pixelate an RGB image array by a given integer factor.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (H, W, 3), dtype=uint8.
    factor : int
        Pixelation factor (>=1). The image is block-averaged at this scale
        then upscaled back using nearest neighbor.

    Returns
    -------
    np.ndarray
        Pixelated image of the same shape and dtype as the input.
    """
    if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("arr must be an RGB image with shape (H, W, 3)")
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return arr.copy()

    padded, orig_hw = _pad_to_multiple(arr, factor)
    H, W, _ = padded.shape

    # Reshape into blocks and average each block
    new_h = H // factor
    new_w = W // factor
    blocks = padded.reshape(new_h, factor, new_w, factor, 3)
    small = blocks.mean(axis=(1, 3)).astype(np.float32)

    # Upscale back using nearest neighbor via repeat
    up = np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)
    up = np.clip(np.rint(up), 0, 255).astype(np.uint8)

    # Crop back to original size if we padded
    oh, ow = orig_hw
    return up[:oh, :ow, :]


def downscale_block_average(arr: Array, factor: int) -> Array:
    """Downscale an RGB image by integer ``factor`` via nearest-neighbor sampling.

    This returns a smaller image of shape (floor(H/f), floor(W/f), 3). It pads
    the input to a multiple of ``factor`` using edge pixels so we can pick one
    representative pixel from each block using nearest-neighbor decimation, then
    crops to the exact floor target size for clean pixel-art downscaling.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (H, W, 3), dtype=uint8.
    factor : int
        Downscale factor (>=1).

    Returns
    -------
    np.ndarray
        Downscaled image array.
    """
    if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("arr must be an RGB image with shape (H, W, 3)")
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return arr.copy()

    H, W, _ = arr.shape
    target_h = H // factor
    target_w = W // factor
    if target_h == 0 or target_w == 0:
        raise ValueError("factor too large for image dimensions")

    padded, _ = _pad_to_multiple(arr, factor)
    Hp, Wp, _ = padded.shape
    # Nearest-neighbor decimation: pick the top-left pixel of each fxf block
    small = padded[0:Hp:factor, 0:Wp:factor, :]
    small = small.astype(np.uint8, copy=False)
    return small[:target_h, :target_w, :]
