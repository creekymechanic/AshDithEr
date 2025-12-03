"""Process a video into a PNG frame sequence using the AshDithEr pipeline.

Reads frames via OpenCV, converts to RGB NumPy arrays, applies the same
image pipeline (downscale → x2 → dither at a chosen stage/scale → x2 →
optional final scale), and writes PNGs into an output directory.

This is sequential and simple: good for offline batch processing.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image

from .utils.pixelate import downscale_block_average
from .utils.upscale import upscale_nearest
from .utils.resize import resize_nearest_scale
from .dithers import apply_dither


def process_frame(arr: np.ndarray, pixel: int, colors: int, dither: str, stage: str, dither_scale: float, scale_final: int) -> np.ndarray:
    """Apply the configured pipeline to a single RGB frame."""
    img_ds = downscale_block_average(arr, pixel) if pixel > 1 else arr.copy()
    img_up1 = upscale_nearest(img_ds, 2)
    img_up2 = upscale_nearest(img_up1, 2)
    img_final_pre = upscale_nearest(img_up2, scale_final) if scale_final > 1 else img_up2

    def apply_on(base: np.ndarray) -> np.ndarray:
        if dither_scale != 1.0:
            tmp = resize_nearest_scale(base, dither_scale)
            tmp = apply_dither(tmp, num_colors=colors, method=dither)
            return resize_nearest_scale(tmp, 1.0 / dither_scale)
        return apply_dither(base, num_colors=colors, method=dither)

    if dither == "none":
        # Skip dithering; still follow upscales
        work = img_up1 if stage == "after-upscale1" else (img_up2 if stage == "after-upscale2" else (img_ds if stage == "after-downscale" else img_final_pre))
    elif stage == "after-downscale":
        work = apply_on(img_ds)
        work = upscale_nearest(work, 2)
        work = upscale_nearest(work, 2)
        if scale_final > 1:
            work = upscale_nearest(work, scale_final)
    elif stage == "after-upscale1":
        work = apply_on(img_up1)
        work = upscale_nearest(work, 2)
        if scale_final > 1:
            work = upscale_nearest(work, scale_final)
    elif stage == "after-upscale2":
        work = apply_on(img_up2)
        if scale_final > 1:
            work = upscale_nearest(work, scale_final)
    else:  # after-final
        work = apply_on(img_final_pre)
    return work


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process video frames to PNG using AshDithEr pipeline")
    p.add_argument("-i", "--input", required=True, help="Input video path")
    p.add_argument("-o", "--outdir", required=True, help="Output directory for PNG frames")
    p.add_argument("--start", type=int, default=0, help="Start frame index (default 0)")
    p.add_argument("--end", type=int, default=None, help="End frame index (exclusive). Default: till end")

    # Pipeline args mirroring the CLI
    p.add_argument("--pixel", type=int, default=1, help="Downscale factor (>=1)")
    p.add_argument("--colors", type=int, default=256, help="Palette size (>=2)")
    p.add_argument("--dither", type=str, default="none", choices=[
        "none", "floyd", "atkinson", "burkes", "sierra", "bayer2", "bayer4", "bayer8"
    ])
    p.add_argument("--dither-stage", type=str, default="after-upscale1", choices=[
        "after-downscale", "after-upscale1", "after-upscale2", "after-final"
    ])
    p.add_argument("--dither-scale", type=float, default=1.0, help="Scaled-resolution dithering factor")
    p.add_argument("--scale", type=int, default=1, help="Final extra upscale (>=1)")

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Failed to open video: {args.input}")
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    idx = 0
    written = 0
    start = int(args.start)
    end = None if args.end is None else int(args.end)

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx < start:
            idx += 1
            continue
        if end is not None and idx >= end:
            break

        # Convert BGR->RGB uint8
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Process
        out = process_frame(
            frame_rgb,
            pixel=max(1, args.pixel),
            colors=max(2, args.colors),
            dither=args.dither,
            stage=args.dither_stage,
            dither_scale=float(args.dither_scale),
            scale_final=max(1, args.scale),
        )

        # Save as PNG: frame_{:06d}.png
        png_path = outdir / f"frame_{idx:06d}.png"
        Image.fromarray(out, mode="RGB").save(png_path)
        written += 1
        idx += 1

    cap.release()
    print(f"Wrote {written} frames to {outdir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
