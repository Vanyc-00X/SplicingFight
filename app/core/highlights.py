from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.signal import find_peaks, medfilt

from app.core.ffmpeg_utils import extract_audio_wav, extract_frame


ProgressCallback = Callable[[float, str], None]


@dataclass
class HighlightCandidate:
    start_sec: float
    end_sec: float
    score: float
    kind: str
    thumbnail_path: str | None = None
    notes: str | None = None


def _iter_sampled_frames(
    video_path: str | Path,
    *,
    sample_fps: float,
    max_seconds: float | None,
):
    """Yield (time_sec, frame_idx, frame) using grab() to skip decode work."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_seconds is not None and src_fps > 0:
        total = min(total, int(max_seconds * src_fps))
    step = max(int(round(src_fps / max(sample_fps, 0.5))), 1)
    idx = 0
    try:
        while True:
            if max_seconds is not None and idx / src_fps > max_seconds:
                break
            if total and idx >= total:
                break
            if idx % step != 0:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            yield idx / src_fps, idx, frame, total
            idx += 1
    finally:
        cap.release()


def _motion_scores(
    video_path: str | Path,
    *,
    sample_fps: float = 8.0,
    max_seconds: float | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    scores: list[float] = []
    prev_gray = None
    for t, idx, frame, total in _iter_sampled_frames(
        video_path, sample_fps=sample_fps, max_seconds=max_seconds
    ):
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(np.mean(diff))
        else:
            score = 0.0
        prev_gray = gray
        times.append(t)
        scores.append(score)
        if progress and total:
            progress(min(0.55 * idx / max(total, 1), 0.55), "Анализ движения…")
    return np.asarray(times), np.asarray(scores, dtype=np.float32)


def _pose_activity_scores(
    video_path: str | Path,
    *,
    sample_fps: float = 3.0,
    max_seconds: float | None = None,
    progress: ProgressCallback | None = None,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose velocity via YOLO pose on CUDA (batched FP16); empty if model fails."""
    try:
        from app.core.device import torch_device, ultralytics_kwargs
        from app.core.model_cache import get_yolo

        model = get_yolo("yolov8n-pose.pt")
        imgsz = 448 if torch_device() == "cuda" else 512
        pred_kw = ultralytics_kwargs(imgsz=imgsz)
        if torch_device() == "cuda":
            batch_size = max(batch_size, 12)
    except Exception:
        return np.array([]), np.array([])

    times: list[float] = []
    scores: list[float] = []
    prev_kps = None
    batch_frames: list[np.ndarray] = []
    batch_times: list[float] = []
    batch_idxs: list[int] = []
    last_total = 1

    def flush() -> None:
        nonlocal prev_kps
        if not batch_frames:
            return
        results = model.predict(batch_frames, conf=0.25, **pred_kw)
        for i, result in enumerate(results):
            velocity = 0.0
            if result.keypoints is not None and len(result.keypoints) > 0:
                kps = result.keypoints.xy[0].cpu().numpy()
                if prev_kps is not None and prev_kps.shape == kps.shape:
                    velocity = float(np.linalg.norm(kps - prev_kps, axis=1).mean())
                prev_kps = kps
            times.append(batch_times[i])
            scores.append(velocity)
            if progress and last_total:
                progress(
                    0.55 + 0.25 * min(batch_idxs[i] / last_total, 1.0),
                    "Анализ поз (CUDA)…",
                )
        batch_frames.clear()
        batch_times.clear()
        batch_idxs.clear()

    try:
        for t, idx, frame, total in _iter_sampled_frames(
            video_path, sample_fps=sample_fps, max_seconds=max_seconds
        ):
            last_total = total or last_total
            # Downscale before GPU — less PCIe transfer
            h, w = frame.shape[:2]
            if max(h, w) > 960:
                scale = 960 / max(h, w)
                frame = cv2.resize(
                    frame,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            batch_frames.append(frame)
            batch_times.append(t)
            batch_idxs.append(idx)
            if len(batch_frames) >= batch_size:
                flush()
        flush()
    except Exception:
        return np.array([]), np.array([])

    return np.asarray(times), np.asarray(scores, dtype=np.float32)


def _audio_scores(video_path: str | Path, cache_wav: str | Path, progress: ProgressCallback | None = None) -> tuple[np.ndarray, np.ndarray]:
    try:
        import soundfile as sf
    except Exception:
        return np.array([]), np.array([])

    wav = Path(cache_wav)
    try:
        if not wav.exists():
            if progress:
                progress(0.82, "Извлечение аудио…")
            extract_audio_wav(video_path, wav)
        audio, sr = sf.read(str(wav), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        win = max(int(sr * 0.25), 1)
        hop = max(int(sr * 0.125), 1)
        times = []
        scores = []
        for start in range(0, max(len(audio) - win, 1), hop):
            chunk = audio[start : start + win]
            rms = float(np.sqrt(np.mean(chunk**2) + 1e-8))
            times.append(start / sr)
            scores.append(rms)
        if progress:
            progress(0.9, "Анализ звука зала…")
        return np.asarray(times), np.asarray(scores, dtype=np.float32)
    except Exception:
        return np.array([]), np.array([])


def _normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _resample_to(times_src: np.ndarray, values: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    if times_src.size == 0 or values.size == 0:
        return np.zeros_like(times_dst, dtype=np.float32)
    return np.interp(times_dst, times_src, values).astype(np.float32)


def detect_highlights(
    video_path: str | Path,
    *,
    cache_dir: str | Path,
    clip_pad_before: float = 1.5,
    clip_pad_after: float = 2.5,
    min_clip: float = 5.0,
    max_clip: float = 12.0,
    max_candidates: int = 20,
    max_seconds: float | None = None,
    use_pose: bool = True,
    progress: ProgressCallback | None = None,
) -> list[HighlightCandidate]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    t_motion, motion = _motion_scores(
        video_path, max_seconds=max_seconds, progress=progress
    )
    if t_motion.size == 0:
        return []

    if use_pose:
        t_pose, pose = _pose_activity_scores(
            video_path, max_seconds=max_seconds, progress=progress
        )
    else:
        t_pose, pose = np.array([]), np.array([])
    t_audio, audio = _audio_scores(video_path, cache_dir / "audio.wav", progress=progress)

    motion_n = _normalize(motion)
    pose_n = _normalize(_resample_to(t_pose, pose, t_motion)) if pose.size else np.zeros_like(motion_n)
    audio_n = _normalize(_resample_to(t_audio, audio, t_motion)) if audio.size else np.zeros_like(motion_n)

    # Combined score: motion + pose + audio boost
    score = 0.55 * motion_n + 0.25 * pose_n + 0.20 * audio_n
    if score.size >= 5:
        score = medfilt(score, kernel_size=5)

    # Peaks for attacks / contact
    min_dist = max(int(2.0 / max((t_motion[1] - t_motion[0]) if t_motion.size > 1 else 0.2, 1e-3)), 1)
    peaks, _props = find_peaks(score, height=0.45, distance=min_dist)
    if peaks.size == 0:
        peaks, _props = find_peaks(score, height=0.30, distance=min_dist)
    if peaks.size == 0:
        # Always offer top activity windows so SMM has something to edit
        top_n = min(max_candidates, max(3, score.size // 8))
        peaks = np.argsort(score)[-top_n:]
        peaks = np.sort(peaks)

    duration = float(t_motion[-1]) if t_motion.size else 0
    candidates: list[HighlightCandidate] = []

    for peak_idx in peaks.tolist():
        peak_t = float(t_motion[peak_idx])
        peak_score = float(score[peak_idx])
        audio_boost = float(audio_n[peak_idx]) if audio_n.size else 0
        pose_boost = float(pose_n[peak_idx]) if pose_n.size else 0

        kind = "action"
        if audio_boost > 0.7 and peak_score > 0.5:
            kind = "crowd_reaction"
        elif pose_boost > 0.65:
            kind = "strike"
        elif peak_score > 0.75:
            kind = "contact"

        start = max(0.0, peak_t - clip_pad_before)
        end = min(duration, peak_t + clip_pad_after)
        length = end - start
        if length < min_clip:
            extra = (min_clip - length) / 2
            start = max(0.0, start - extra)
            end = min(duration, end + extra)
        if end - start > max_clip:
            end = start + max_clip

        # merge overlap later
        candidates.append(
            HighlightCandidate(
                start_sec=round(start, 2),
                end_sec=round(end, 2),
                score=round(peak_score + 0.15 * audio_boost, 3),
                kind=kind,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[:max_candidates]
    candidates = _merge_overlapping(candidates)
    candidates.sort(key=lambda c: c.start_sec)

    # thumbnails
    thumb_dir = cache_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(candidates):
        mid = (c.start_sec + c.end_sec) / 2
        thumb = thumb_dir / f"hl_{i}_{int(mid * 1000)}.jpg"
        try:
            extract_frame(video_path, mid, thumb)
            c.thumbnail_path = str(thumb)
        except Exception:
            c.thumbnail_path = None
        if progress:
            progress(0.9 + 0.1 * (i + 1) / max(len(candidates), 1), "Превью моментов…")

    if progress:
        progress(1.0, f"Найдено кандидатов: {len(candidates)}")
    return candidates


def _merge_overlapping(items: list[HighlightCandidate], gap: float = 1.0) -> list[HighlightCandidate]:
    if not items:
        return []
    ordered = sorted(items, key=lambda c: c.start_sec)
    merged = [ordered[0]]
    for cur in ordered[1:]:
        prev = merged[-1]
        if cur.start_sec <= prev.end_sec + gap:
            prev.end_sec = max(prev.end_sec, cur.end_sec)
            prev.score = max(prev.score, cur.score)
            if cur.score >= prev.score:
                prev.kind = cur.kind
        else:
            merged.append(cur)
    return merged
