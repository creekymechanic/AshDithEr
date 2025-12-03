"""Ultra-simple PNG sequence to MP4 stitcher with auto FPS and audio.

Usage (one-liners):
    python -m AshDithEr.stitch -f C:\frames --source C:\in.mp4
    python -m AshDithEr.stitch -f C:\frames

Behavior:
- Picks FPS from --source video if provided; else defaults to 30.
- Remuxes audio from --source if provided; else no audio.
- Accepts frames named like frame_000000.png, or any *.png via glob.
- Writes output next to frames dir as frames.mp4 unless --output is set.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import ffmpeg  # type: ignore


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stitch PNG frames to MP4 with auto FPS/audio")
    p.add_argument("-f", "--frames", required=True, help="Directory with PNG frames")
    p.add_argument("--source", type=str, default=None, help="Optional source video to derive FPS and audio")
    p.add_argument("--output", type=str, default=None, help="Output MP4 path (default: <frames>/frames.mp4)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")
    return p.parse_args(argv)


def _probe_fps(src: str) -> Optional[float]:
    try:
        info = ffmpeg.probe(src)
    except Exception:
        return None
    # Find video stream and parse r_frame_rate or avg_frame_rate
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    for s in vstreams:
        rate = s.get("avg_frame_rate") or s.get("r_frame_rate")
        if rate and rate != "0/0":
            try:
                num, den = rate.split("/")
                num = float(num)
                den = float(den) if float(den) != 0 else 1.0
                return num / den
            except Exception:
                continue
    return None


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    frames_dir = Path(args.frames)
    if not frames_dir.exists():
        print(f"Frames directory not found: {frames_dir}")
        return 2

    out_path = Path(args.output) if args.output else frames_dir / "frames.mp4"
    if out_path.exists() and not args.overwrite:
        print(f"Output exists (use --overwrite): {out_path}")
        return 2

    fps = 30.0
    a_in = None
    if args.source:
        fps_probe = _probe_fps(args.source)
        if fps_probe and fps_probe > 0:
            fps = fps_probe
        try:
            a_in = ffmpeg.input(args.source)
        except Exception:
            a_in = None

    # Choose input: prefer numeric pattern, else glob
    numeric_pattern = os.path.join(str(frames_dir), "frame_%06d.png")
    glob_pattern = os.path.join(str(frames_dir), "*.png")

    # If at least one numeric file exists, use numeric; else glob
    has_numeric = any((frames_dir / (f"frame_{i:06d}.png")).exists() for i in range(0, 1)) or any(p.name.startswith("frame_") and p.suffix.lower() == ".png" for p in frames_dir.iterdir())

    if has_numeric:
        v_in = ffmpeg.input(numeric_pattern, framerate=fps)
    else:
        v_in = ffmpeg.input(glob_pattern, pattern_type="glob", framerate=fps, safe=0)

    out_kwargs = dict(vcodec="libx264", pix_fmt="yuv420p", crf=18, preset="medium", movflags="+faststart")

    if a_in is not None:
        stream = ffmpeg.output(v_in, a_in.audio, str(out_path), acodec="copy", **out_kwargs)
    else:
        stream = ffmpeg.output(v_in, str(out_path), **out_kwargs)

    if args.overwrite:
        stream = ffmpeg.overwrite_output(stream)

    try:
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as ex:  # type: ignore
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
