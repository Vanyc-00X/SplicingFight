from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.crop import PRESETS, crop_at_time
from app.core.ffmpeg_utils import ffmpeg_bin, run_ffmpeg
from app.core.tracking import TrackPoint, load_tracking


ProgressCallback = Callable[[float, str], None]


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ):
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _draw_overlay(
    frame_bgr: np.ndarray,
    *,
    title: str | None,
    result: str | None,
    logo_path: str | None,
    brand_color: str,
    text_style: str,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    title_size = 54 if text_style == "bold" else 42
    result_size = 40 if text_style == "bold" else 32
    font_title = _load_font(title_size)
    font_result = _load_font(result_size)

    color = brand_color if brand_color.startswith("#") else f"#{brand_color}"
    margin = int(h * 0.04)

    if title:
        # bottom banner
        banner_h = int(h * 0.12)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([(0, h - banner_h), (w, h)], fill=(0, 0, 0, 150))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((margin, h - banner_h + margin // 2), title, fill="white", font=font_title)
        if result:
            draw.text((margin, h - banner_h + margin // 2 + title_size + 4), result, fill=color, font=font_result)

    if logo_path and Path(logo_path).exists():
        try:
            logo = Image.open(logo_path).convert("RGBA")
            lw = int(w * 0.14)
            ratio = lw / logo.width
            logo = logo.resize((lw, max(int(logo.height * ratio), 1)))
            img = img.convert("RGBA")
            img.paste(logo, (w - logo.width - margin, margin), logo)
            img = img.convert("RGB")
        except Exception:
            pass

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_clip(
    source: str | Path,
    output: str | Path,
    *,
    start_sec: float,
    end_sec: float,
    preset: str,
    tracking_points: list[TrackPoint] | None = None,
    tracking_json: str | None = None,
    frame_w: int,
    frame_h: int,
    title: str | None = None,
    result: str | None = None,
    logo_path: str | None = None,
    brand_color: str = "#C62828",
    text_style: str = "bold",
    music_path: str | None = None,
    music_volume: float = 0.35,
    muted: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    preset_cfg = PRESETS.get(preset, PRESETS["reels"])
    out_w, out_h = preset_cfg["width"], preset_cfg["height"]
    aspect = preset_cfg["aspect"]
    points = tracking_points if tracking_points is not None else load_tracking(tracking_json)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    start_f = int(start_sec * fps)
    end_f = max(int(end_sec * fps), start_f + 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    tmp_dir = Path(tempfile.mkdtemp(prefix="sf_render_"))
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    total = end_f - start_f
    written = 0
    idx = start_f
    while idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        rect = crop_at_time(points, t, frame_w, frame_h, aspect)
        x, y, cw, ch = rect.x, rect.y, rect.w, rect.h
        crop = frame[y : y + ch, x : x + cw]
        if crop.size == 0:
            crop = frame
        resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if title or logo_path:
            resized = _draw_overlay(
                resized,
                title=title,
                result=result,
                logo_path=logo_path,
                brand_color=brand_color,
                text_style=text_style,
            )
        cv2.imwrite(str(frames_dir / f"f_{written:06d}.jpg"), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        written += 1
        idx += 1
        if progress and total:
            progress(min(0.85 * written / total, 0.85), f"Рендер кадров {written}/{total}")
    cap.release()

    if written == 0:
        raise RuntimeError("Нет кадров для экспорта")

    silent_video = tmp_dir / "silent.mp4"
    run_ffmpeg(
        [
            "-framerate",
            f"{fps:.3f}",
            "-i",
            str(frames_dir / "f_%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(silent_video),
        ]
    )

    # audio mix
    duration = written / fps
    args: list[str] = ["-i", str(silent_video)]
    filter_parts: list[str] = []
    map_audio = None

    if not muted:
        args += ["-ss", f"{start_sec:.3f}", "-t", f"{duration:.3f}", "-i", str(source)]
        if music_path and Path(music_path).exists():
            args += ["-stream_loop", "-1", "-i", str(music_path)]
            filter_parts.append(
                f"[1:a]volume=1.0[a1];[2:a]volume={music_volume:.3f},atrim=0:{duration:.3f},asetpts=PTS-STARTPTS[a2];"
                f"[a1][a2]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            map_audio = "[aout]"
        else:
            map_audio = "1:a?"
    elif music_path and Path(music_path).exists():
        args += ["-stream_loop", "-1", "-i", str(music_path)]
        filter_parts.append(
            f"[1:a]volume={music_volume:.3f},atrim=0:{duration:.3f},asetpts=PTS-STARTPTS[aout]"
        )
        map_audio = "[aout]"

    out_args = ["-map", "0:v"]
    if filter_parts:
        out_args += ["-filter_complex", "".join(filter_parts), "-map", map_audio or "1:a?"]
    elif map_audio:
        out_args += ["-map", map_audio]
    out_args += [
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-t",
        f"{duration:.3f}",
        str(output),
    ]

    if progress:
        progress(0.92, "Сборка звука…")
    try:
        run_ffmpeg(args + out_args)
    except Exception:
        # fallback silent
        Path(output).write_bytes(silent_video.read_bytes()) if False else None
        run_ffmpeg(["-i", str(silent_video), "-c", "copy", str(output)])

    if progress:
        progress(1.0, f"Готово: {output.name}")
    return output


def concat_clips(
    clip_paths: list[Path],
    output: Path,
    *,
    fade: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    if not clip_paths:
        raise RuntimeError("Нет клипов для склейки")
    if len(clip_paths) == 1:
        output.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(["-i", str(clip_paths[0]), "-c", "copy", str(output)])
        return output

    list_file = output.parent / f"{output.stem}_concat.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in clip_paths),
        encoding="utf-8",
    )
    if progress:
        progress(0.5, "Склейка клипов…")
    # re-encode for safety across clips
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(output),
        ]
    )
    if progress:
        progress(1.0, "Склейка готова")
    return output


def ffmpeg_available() -> bool:
    try:
        return bool(ffmpeg_bin())
    except Exception:
        return False
