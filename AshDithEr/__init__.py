from __future__ import annotations

# Alias package: re-export public API from the existing implementation.
from pixforge.dithers import apply_dither  # noqa: F401
from pixforge.utils.loader import load_image, save_image  # noqa: F401
from pixforge.utils.pixelate import pixelate, downscale_block_average  # noqa: F401
from pixforge.utils.upscale import upscale_nearest  # noqa: F401
from pixforge.utils.resize import resize_nearest, resize_nearest_scale  # noqa: F401

__all__ = [
    "apply_dither",
    "load_image",
    "save_image",
    "pixelate",
    "downscale_block_average",
    "upscale_nearest",
    "resize_nearest",
    "resize_nearest_scale",
]
