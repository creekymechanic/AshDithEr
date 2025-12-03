"""Command-line entry point for AshDithEr.

This tool loads an image, pixelates it, optionally upscales it,
reduces it to a limited color palette, applies a selected dithering
algorithm, and saves the result.

All processing occurs on NumPy arrays; Pillow is used only for
loading and saving.

Usage example:
    python -m AshDithEr.main -i input.png -o output.png --pixel 4 --scale 2 --dither floyd --colors 32
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from .utils.loader import load_image, save_image
from .utils.pixelate import pixelate, downscale_block_average
from .utils.upscale import upscale_nearest
from .utils.resize import resize_nearest_scale
from .dithers import apply_dither


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Optional list of arguments for testing. If None, uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="AshDithEr",
        description=(
            "Pixelate, upscale, and dither images using several algorithms. "
            "All processing uses NumPy arrays for a clean, pure-Python pipeline."
        ),
    )

    parser.add_argument("-i", "--input", required=True, help="Path to input image file")
    parser.add_argument("-o", "--output", required=True, help="Path to output image file")

    parser.add_argument(
        "--pixel",
        type=int,
        default=1,
        help=(
            "Downscale factor (>=1). Pipeline: downscale by this factor, then x2, "
            "dither, then x2."
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help=(
            "Optional extra final upscale factor (>=1) after the second x2 step."
        ),
    )
    parser.add_argument(
        "--dither",
        type=str,
        default="none",
        choices=[
            "none",
            "floyd",
            "atkinson",
            "burkes",
            "sierra",
            "bayer2",
            "bayer4",
            "bayer8",
        ],
        help=(
            "Dithering method: none | floyd | atkinson | burkes | sierra | "
            "bayer2 | bayer4 | bayer8"
        ),
    )
    parser.add_argument(
        "--dither-stage",
        type=str,
        default="after-upscale1",
        choices=[
            "after-downscale",
            "after-upscale1",
            "after-upscale2",
            "after-final",
        ],
        help=(
            "When to apply dithering relative to the pipeline: "
            "after-downscale | after-upscale1 | after-upscale2 | after-final. "
            "Default: after-upscale1 (downscale -> x2 -> dither -> x2)."
        ),
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=256,
        help="Target palette size (total colors, e.g., 2..256).",
    )
    parser.add_argument(
        "--dither-scale",
        type=float,
        default=1.0,
        help=(
            "Apply dithering at a scaled resolution: >1 for coarser (larger dots), "
            "<1 for finer (smaller dots). Uses nearest scaling around the chosen stage."
        ),
    )

    return parser.parse_args(argv)


def validate_args(ns: argparse.Namespace) -> None:
    """Validate argument values and raise ValueError for invalid inputs.

    Parameters
    ----------
    ns : argparse.Namespace
        Parsed CLI arguments.
    """
    if ns.pixel < 1:
        raise ValueError("--pixel must be an integer >= 1")
    if ns.scale < 1:
        raise ValueError("--scale must be an integer >= 1")
    if ns.colors < 2:
        raise ValueError("--colors must be >= 2")
    if not Path(ns.input).exists():
        raise ValueError(f"Input file not found: {ns.input}")


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry function for the CLI.

    Parameters
    ----------
    argv : list[str] | None
        Optional list of arguments for testing.

    Returns
    -------
    int
        Exit status code (0 for success, non-zero for failure).
    """
    args = parse_args(argv)
    try:
        validate_args(args)
    except Exception as e:  # pragma: no cover - simple guard
        print(f"Argument error: {e}")
        return 2

    # 1) Load (Pillow -> NumPy RGB uint8)
    img = load_image(args.input)

    # Revised pipeline with flexible dither stage:
    # - Downscale (block-average) by pixel factor
    # - Upscale x2
    # - Dither (configurable stage)
    # - Upscale x2
    # - Optional final upscale

    # 2) Downscale cleanly (block-average) by pixel factor
    img_ds = downscale_block_average(img, args.pixel) if args.pixel > 1 else img.copy()
    # 3) First x2 upscale
    img_up1 = upscale_nearest(img_ds, 2)
    # 4) Second x2 upscale (prepared; may be used depending on stage)
    img_up2 = upscale_nearest(img_up1, 2)
    # 5) Optional extra final upscale (prepared)
    img_final_pre = upscale_nearest(img_up2, args.scale) if args.scale > 1 else img_up2

    stage = args.dither_stage
    d_scale = float(args.dither_scale)
    if stage == "after-downscale":
        base = img_ds
        if d_scale != 1.0:
            tmp = resize_nearest_scale(base, d_scale)
            tmp = apply_dither(tmp, num_colors=args.colors, method=args.dither)
            work = resize_nearest_scale(tmp, 1.0 / d_scale)
        else:
            work = apply_dither(base, num_colors=args.colors, method=args.dither)
        work = upscale_nearest(work, 2)
        work = upscale_nearest(work, 2)
        if args.scale > 1:
            work = upscale_nearest(work, args.scale)
    elif stage == "after-upscale1":
        base = img_up1
        if d_scale != 1.0:
            tmp = resize_nearest_scale(base, d_scale)
            tmp = apply_dither(tmp, num_colors=args.colors, method=args.dither)
            work = resize_nearest_scale(tmp, 1.0 / d_scale)
        else:
            work = apply_dither(base, num_colors=args.colors, method=args.dither)
        work = upscale_nearest(work, 2)
        if args.scale > 1:
            work = upscale_nearest(work, args.scale)
    elif stage == "after-upscale2":
        base = img_up2
        if d_scale != 1.0:
            tmp = resize_nearest_scale(base, d_scale)
            tmp = apply_dither(tmp, num_colors=args.colors, method=args.dither)
            work = resize_nearest_scale(tmp, 1.0 / d_scale)
        else:
            work = apply_dither(base, num_colors=args.colors, method=args.dither)
        if args.scale > 1:
            work = upscale_nearest(work, args.scale)
    elif stage == "after-final":
        base = img_final_pre
        if d_scale != 1.0:
            tmp = resize_nearest_scale(base, d_scale)
            tmp = apply_dither(tmp, num_colors=args.colors, method=args.dither)
            work = resize_nearest_scale(tmp, 1.0 / d_scale)
        else:
            work = apply_dither(base, num_colors=args.colors, method=args.dither)
    else:  # Defensive fallback
        work = apply_dither(img_up1, num_colors=args.colors, method=args.dither)
        work = upscale_nearest(work, 2)
        if args.scale > 1:
            work = upscale_nearest(work, args.scale)

    # 6) Save (NumPy -> Pillow)
    save_image(work, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
