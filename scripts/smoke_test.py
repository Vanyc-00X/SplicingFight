"""Offline smoke test for core pipeline (no GUI)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.crop import crop_at_time, smart_crop_rect
from app.core.export import render_clip
from app.core.ffmpeg_utils import ffmpeg_bin, probe_video
from app.core.highlights import detect_highlights
from app.core.tracking import AthleteTracker, TrackPoint, box_at_time, save_tracking, load_tracking
from app.db.database import Database
from app.paths import ensure_dirs


def make_fake_video(path: Path, seconds: float = 4.0, fps: int = 15) -> None:
    w, h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    frames = int(seconds * fps)
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        # moving "athlete" box
        x = int(40 + i * 3) % (w - 80)
        y = 120 + int(20 * np.sin(i / 5))
        # burst motion mid-video
        size = 50 + (40 if 25 < i < 40 else 0)
        cv2.rectangle(frame, (x, y), (x + size, y + size + 40), (40, 40, 200), -1)
        cv2.circle(frame, (x + size // 2, y + 10), 12, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()


def main() -> None:
    ensure_dirs()
    print("ffmpeg:", ffmpeg_bin())

    with tempfile.TemporaryDirectory(prefix="sf_smoke_") as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "fight.mp4"
        make_fake_video(video)
        info = probe_video(video)
        print("probe:", info)
        assert info["duration"] > 0
        assert info["width"] == 640

        db_path = tmp_path / "test.db"
        db = Database(db_path)
        pid = db.create_project("Smoke Fight", str(video), **info)
        assert db.get_project(pid)["name"] == "Smoke Fight"

        # Tracking with synthetic points (avoid heavy YOLO download in CI-less smoke if needed)
        # Still try one YOLO detect; if it fails, use synthetic track.
        points: list[TrackPoint]
        try:
            tracker = AthleteTracker()
            cap = cv2.VideoCapture(str(video))
            ok, frame = cap.read()
            cap.release()
            boxes = tracker.detect_people(frame) if ok else []
            if boxes:
                box = boxes[0][:4]
                points = tracker.track_video(video, box, sample_fps=5)
                print("yolo track points:", len(points))
            else:
                raise RuntimeError("no people in synthetic video")
        except Exception as exc:
            print("YOLO fallback synthetic track:", exc)
            points = [
                TrackPoint(t=i * 0.2, x=40 + i * 8, y=120, w=60, h=90, conf=0.9)
                for i in range(20)
            ]

        track_json = tmp_path / "tracking.json"
        save_tracking(track_json, points)
        loaded = load_tracking(track_json)
        assert box_at_time(loaded, 1.0) is not None
        db.update_project(pid, tracking_json=str(track_json))

        rect = smart_crop_rect(640, 360, (100, 100, 60, 90), 9 / 16)
        assert rect.w > 0 and rect.h > 0
        c = crop_at_time(loaded, 0.5, 640, 360, 16 / 9)
        assert c.w > 0

        cache = tmp_path / "cache"
        highlights = detect_highlights(video, cache_dir=cache, max_candidates=10)
        print("highlights:", len(highlights), [ (h.kind, h.score) for h in highlights[:5] ])
        db.replace_highlights(
            pid,
            [
                {
                    "start_sec": h.start_sec,
                    "end_sec": h.end_sec,
                    "score": h.score,
                    "kind": h.kind,
                    "thumbnail_path": h.thumbnail_path,
                    "accepted": 1,
                }
                for h in highlights[:3]
            ]
            or [
                {
                    "start_sec": 0.5,
                    "end_sec": 2.5,
                    "score": 1.0,
                    "kind": "manual",
                    "accepted": 1,
                }
            ],
        )
        hls = db.list_highlights(pid)
        assert hls

        out = tmp_path / "out_reels.mp4"
        h0 = hls[0]
        render_clip(
            video,
            out,
            start_sec=float(h0["start_sec"]),
            end_sec=float(h0["end_sec"]),
            preset="reels",
            tracking_points=loaded,
            frame_w=640,
            frame_h=360,
            title="Smoke Test",
            result="Ippon",
            brand_color="#C62828",
        )
        assert out.exists() and out.stat().st_size > 1000
        print("exported:", out, out.stat().st_size)

        db.add_library_clip(
            project_id=pid,
            title="Smoke clip",
            athlete="Tester",
            format="reels",
            output_path=str(out),
        )
        assert db.list_library()
        job = db.create_render_job(pid, "youtube", {"mode": "single", "start_sec": 0.5, "end_sec": 2.0})
        assert db.next_queued_job()["id"] == job
        print("SMOKE OK")


if __name__ == "__main__":
    main()
