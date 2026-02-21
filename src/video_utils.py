from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def list_videos(video_dir: str) -> list[str]:
    root = Path(video_dir)
    return [str(p) for p in root.iterdir() if p.suffix.lower() in VIDEO_EXTS]


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _duration_seconds(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def iter_video_clips(
    video_path: str,
    clip_seconds: int = 15,
    stride_seconds: int = 30,
) -> Iterable[str]:
    if not _ffmpeg_available():
        yield video_path
        return

    duration = _duration_seconds(video_path)
    if duration <= clip_seconds:
        yield video_path
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        start = 0.0
        index = 0
        while start < duration:
            out_path = os.path.join(tmpdir, f"clip_{index:04d}.mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-i",
                video_path,
                "-t",
                str(clip_seconds),
                "-c",
                "copy",
                out_path,
            ]
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            yield out_path
            start += stride_seconds
            index += 1
