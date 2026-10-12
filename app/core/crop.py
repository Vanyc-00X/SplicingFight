from __future__ import annotations

from dataclasses import dataclass

from app.core.tracking import TrackPoint, box_at_time


@dataclass
class CropRect:
    x: int
    y: int
    w: int
    h: int


PRESETS = {
    "reels": {"width": 1080, "height": 1920, "aspect": 9 / 16, "label": "Reels 9:16"},
    "youtube": {"width": 1920, "height": 1080, "aspect": 16 / 9, "label": "YouTube 16:9"},
}


def smart_crop_rect(
    frame_w: int,
    frame_h: int,
    box: tuple[float, float, float, float] | None,
    aspect: float,
    *,
    padding: float = 1.8,
) -> CropRect:
    """Compute axis-aligned crop of given aspect around athlete box."""
    if box is None:
        # center crop
        if frame_w / max(frame_h, 1) > aspect:
            h = frame_h
            w = int(h * aspect)
        else:
            w = frame_w
            h = int(w / aspect)
        return CropRect(x=max((frame_w - w) // 2, 0), y=max((frame_h - h) // 2, 0), w=w, h=h)

    x, y, bw, bh = box
    cx = x + bw / 2
    cy = y + bh / 2
    target_h = max(bh * padding, frame_h * 0.45)
    target_w = target_h * aspect
    if target_w > frame_w:
        target_w = float(frame_w)
        target_h = target_w / aspect
    if target_h > frame_h:
        target_h = float(frame_h)
        target_w = target_h * aspect

    left = int(round(cx - target_w / 2))
    top = int(round(cy - target_h / 2))
    left = max(0, min(left, frame_w - int(target_w)))
    top = max(0, min(top, frame_h - int(target_h)))
    return CropRect(x=left, y=top, w=int(target_w), h=int(target_h))


def crop_at_time(
    points: list[TrackPoint],
    t: float,
    frame_w: int,
    frame_h: int,
    aspect: float,
) -> CropRect:
    box = box_at_time(points, t)
    return smart_crop_rect(frame_w, frame_h, box, aspect)


def smooth_crop_keyframes(
    points: list[TrackPoint],
    frame_w: int,
    frame_h: int,
    aspect: float,
) -> list[tuple[float, CropRect]]:
    keyframes: list[tuple[float, CropRect]] = []
    for p in points:
        rect = smart_crop_rect(frame_w, frame_h, (p.x, p.y, p.w, p.h), aspect)
        if keyframes:
            prev_t, prev = keyframes[-1]
            # light smoothing
            alpha = 0.35
            rect = CropRect(
                x=int(prev.x * (1 - alpha) + rect.x * alpha),
                y=int(prev.y * (1 - alpha) + rect.y * alpha),
                w=int(prev.w * (1 - alpha) + rect.w * alpha),
                h=int(prev.h * (1 - alpha) + rect.h * alpha),
            )
        keyframes.append((p.t, rect))
    return keyframes
