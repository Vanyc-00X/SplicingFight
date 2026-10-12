from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg


def ffmpeg_bin() -> str:
    system = shutil.which("ffmpeg")
    if system:
        return system
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_bin() -> str | None:
    system = shutil.which("ffprobe")
    if system:
        return system
    # imageio-ffmpeg may ship only ffmpeg; fall back to ffmpeg -i parsing via OpenCV
    return None


def run_ffmpeg(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [ffmpeg_bin(), "-y", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
        encoding="utf-8",
        errors="replace",
    )


def probe_video(path: str | Path) -> dict[str, Any]:
    """Return duration, fps, width, height. Uses ffprobe if available, else OpenCV."""
    path = Path(path)
    probe = ffprobe_bin()
    if probe:
        proc = subprocess.run(
            [
                probe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            video_stream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
                {},
            )
            fmt = data.get("format", {})
            fps = 25.0
            rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "25/1"
            if isinstance(rate, str) and "/" in rate:
                num, den = rate.split("/", 1)
                try:
                    fps = float(num) / max(float(den), 1e-6)
                except ValueError:
                    fps = 25.0
            return {
                "duration": float(fmt.get("duration") or video_stream.get("duration") or 0),
                "fps": fps,
                "width": int(video_stream.get("width") or 0),
                "height": int(video_stream.get("height") or 0),
            }

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0
    return {"duration": duration, "fps": fps, "width": width, "height": height}


def create_proxy(source: str | Path, dest: str | Path, *, height: int = 720) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(source),
            "-vf",
            f"scale=-2:{height}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(dest),
        ]
    )
    return dest


def extract_audio_wav(source: str | Path, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dest),
        ]
    )
    return dest


def extract_frame(source: str | Path, time_sec: float, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-ss",
            f"{max(time_sec, 0):.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(dest),
        ]
    )
    return dest
