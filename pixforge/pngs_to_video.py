"""Stitch a PNG frame sequence into an MP4, with optional audio remux.

Requires ffmpeg available on PATH (ffmpeg.exe) and ffmpeg-python.

Examples
--------
Basic encode (frames like frame_000000.png, frame_000001.png):
    python -m AshDithEr.pngs_to_video -f C:\frames -o C:\out.mp4 --fps 30

Using a custom numeric pattern and start-number:
    python -m AshDithEr.pngs_to_video -f C:\frames -o C:\out.mp4 \
        --pattern "frame_%08d.png" --start-number 12 --fps 24

Glob mode (any *.png in alphabetical order):
    python -m AshDithEr.pngs_to_video -f C:\frames -o C:\out.mp4 --glob --fps 60

Remux audio from original video (copy audio stream):
    python -m AshDithEr.pngs_to_video -f C:\frames -o C:\out.mp4 \
        --audio-src C:\original.mp4 --fps 30
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import ffmpeg  # type: ignore


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stitch PNG sequence into MP4 with optional audio remux")
    p.add_argument("-f", "--frames", required=True, help="Directory containing PNG frames")
    p.add_argument("-o", "--output", required=True, help="Output MP4 path")
    p.add_argument("--fps", type=float, default=30.0, help="Frames per second for the sequence")
    p.add_argument("--pattern", type=str, default="frame_%06d.png", help="Numeric pattern for frames (image2)")
    p.add_argument("--start-number", type=int, default=None, help="Start index for numeric pattern (image2)")
    p.add_argument("--glob", action="store_true", help="Use glob pattern matching (e.g., *.png) instead of numeric pattern")
    p.add_argument("--audio-src", type=str, default=None, help="Optional source video to remux audio from (copied)")
    p.add_argument("--crf", type=int, default=18, help="x264 CRF (lower is higher quality/larger size)")
    p.add_argument("--preset", type=str, default="medium", help="x264 preset: ultrafast..veryslow")
    p.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    frames_dir = Path(args.frames)
    if not frames_dir.exists():
        print(f"Frames directory not found: {frames_dir}")
        return 2

    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        print(f"Output exists (use --overwrite to replace): {out_path}")
        return 2

    try:
        if args.glob:
            pattern = os.path.join(str(frames_dir), "*.png")
            v_in = ffmpeg.input(
                pattern,
                pattern_type="glob",
                framerate=args.fps,
                safe=0,
            )
        else:
            pattern = os.path.join(str(frames_dir), args.pattern)
            kwargs = {"framerate": args.fps}
            if args.start_number is not None:
                kwargs["start_number"] = args.start_number
            v_in = ffmpeg.input(pattern, **kwargs)

        out_kwargs = dict(vcodec="libx264", pix_fmt="yuv420p", crf=args.crf, preset=args.preset, movflags="+faststart")

        if args.audio_src:
            a_in = ffmpeg.input(args.audio_src)
            stream = ffmpeg.output(
                v_in,
                a_in.audio,
                str(out_path),
                acodec="copy",
                shortest=None,  # keep as long as the shortest stream if set to 1
                **out_kwargs,
            )
        else:
            stream = ffmpeg.output(v_in, str(out_path), **out_kwargs)

        if args.overwrite:
            stream = ffmpeg.overwrite_output(stream)

        stream = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as ex:  # type: ignore
        # Show ffmpeg stderr for easier debugging
        print(ex.stderr.decode("utf-8", errors="ignore"))
        return 1
    except FileNotFoundError as ex:
        print("ffmpeg binary not found. Install ffmpeg and ensure it's on PATH.")
        print(str(ex))
        return 1

    print(f"Wrote video: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
