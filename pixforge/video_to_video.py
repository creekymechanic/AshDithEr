"""Direct MP4→MP4 processing using the AshDithEr image pipeline.

Reads an input video, applies the AshDith pipeline to each frame,
and writes an MP4, optionally remuxing original audio.

Examples:
  python -m pixforge.video_to_video --input C:\in.mp4 --output C:\out.mp4 \
    --pixel 16 --colors 32 --dither bayer2 --dither-stage after-upscale2 \
    --dither-scale 1 --final-scale 4 --overwrite

Requires: ffmpeg (binary on PATH), ffmpeg-python, OpenCV (opencv-python).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import numpy as np
import ffmpeg  # type: ignore

from .utils.loader import save_image
from .utils.pixelate import downscale_block_average
from .utils.upscale import upscale_nearest
from .utils.resize import resize_nearest_scale
from .dithers import apply_dither


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process video to video using AshDith pipeline")
    p.add_argument("--input", required=True, help="Input video file (mp4, etc.)")
    p.add_argument("--output", required=True, help="Output MP4 path")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")

    # Pipeline controls (mirrors CLI)
    p.add_argument("--pixel", type=int, default=16, help="Pixel size for initial downscale (nearest)")
    p.add_argument("--colors", type=int, default=32, help="Approx color count / per-channel levels")
    p.add_argument("--dither", type=str, default="bayer2", help="Dither: floyd|atkinson|burkes|sierra|bayer2|bayer4|bayer8")
    p.add_argument(
        "--dither-stage",
        type=str,
        default="after-upscale2",
        choices=["before-upscale", "after-upscale", "after-upscale2"],
        help="When to apply dithering in the pipeline",
    )
    p.add_argument("--dither-scale", type=int, default=1, help="Temporary scaling to control dither grain size")
    p.add_argument("--final-scale", type=int, default=4, help="Final nearest-neighbor scale factor")

    return p.parse_args(argv)


def _probe_audio_stream(src: str) -> bool:
    try:
        info = ffmpeg.probe(src)
    except Exception:
        return False
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def _open_ffmpeg_pipe(out_path: str, width: int, height: int, fps: float):
    in_stream = ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgb24",
        s=f"{width}x{height}",
        framerate=fps,
    )
    out_stream = ffmpeg.output(
        in_stream,
        out_path,
        vcodec="libx264",
        pix_fmt="yuv420p",
        crf=18,
        preset="medium",
        movflags="+faststart",
    )
    out_stream = ffmpeg.overwrite_output(out_stream)
    process = ffmpeg.run_async(out_stream, pipe_stdin=True)
    return process


def _process_frame_rgb(rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    # Pipeline: downscale (nearest via pixel size) → x2 → dither per stage/scale → x2 → final scale
    h, w, _ = rgb.shape
    # Derive downscaled size by pixel size
    ds_h = max(1, h // args.pixel)
    ds_w = max(1, w // args.pixel)
    base = downscale_block_average(rgb, args.pixel)
    # base size is roughly h/pixel, w/pixel

    # First x2 upscale
    up1 = upscale_nearest(base, 2)

    def maybe_dither(arr: np.ndarray) -> np.ndarray:
        # Apply dither with temporary scale to control grain
        if args.dither_scale > 1:
            tmp = resize_nearest_scale(arr, args.dither_scale)
            dit = apply_dither(tmp, args.colors, args.dither)
            return resize_nearest_scale(dit, 1 / args.dither_scale)
        else:
            return apply_dither(arr, args.colors, args.dither)

    if args.dither_stage == "before-upscale":
        d1 = maybe_dither(base)
        up1 = upscale_nearest(d1, 2)
    elif args.dither_stage == "after-upscale":
        up1 = maybe_dither(up1)
    elif args.dither_stage == "after-upscale2":
        up1 = upscale_nearest(up1, 2)
        up1 = maybe_dither(up1)
    else:
        up1 = maybe_dither(up1)

    # Final scale
    if args.final_scale and args.final_scale != 1:
        out = resize_nearest_scale(up1, args.final_scale)
    else:
        out = up1

    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        print(f"Output exists (use --overwrite): {out_path}")
        return 2

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"Failed to open input: {in_path}")
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-compute output size by running one frame through the pipeline if available
    ret, first_bgr = cap.read()
    if not ret:
        print("No frames in input video.")
        cap.release()
        return 2
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    processed_first = _process_frame_rgb(first_rgb, args)
    out_h, out_w = processed_first.shape[:2]

    process = _open_ffmpeg_pipe(str(out_path), out_w, out_h, fps)
    # Write first frame to pipe as RGB24
    process.stdin.write(processed_first.astype(np.uint8).tobytes())

    # Process remaining frames
    frame_idx = 1
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        processed = _process_frame_rgb(rgb, args)
        if processed.shape[0] != out_h or processed.shape[1] != out_w:
            # Resize to expected output size if the pipeline differed
            processed = cv2.resize(processed, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        process.stdin.write(processed.astype(np.uint8).tobytes())
        frame_idx += 1

    # Close pipe and wait for ffmpeg to finish
    process.stdin.close()
    process.wait()
    cap.release()

    # If input has audio, remux it using ffmpeg to the new video (copy audio)
    if _probe_audio_stream(str(in_path)):
        tmp_out = str(out_path)
        # Create a new file with copied audio track
        v_in = ffmpeg.input(tmp_out)
        a_in = ffmpeg.input(str(in_path))
        combined = ffmpeg.output(v_in.video, a_in.audio, str(out_path), vcodec="copy", acodec="copy", movflags="+faststart")
        try:
            ffmpeg.overwrite_output(combined)
            ffmpeg.run(combined, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as ex:  # type: ignore
            print("Audio remux failed; keeping video without audio.")
            print(ex.stderr.decode("utf-8", errors="ignore"))
        except FileNotFoundError:
            print("ffmpeg binary not found for audio remux.")

    print(f"Wrote processed video: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
