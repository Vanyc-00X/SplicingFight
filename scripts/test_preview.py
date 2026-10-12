"""Integration test: open real fight video in VideoPlayer and verify pixmap."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from app.core.export import render_clip
from app.core.ffmpeg_utils import probe_video
from app.core.tracking import TrackPoint
from app.db.database import Database
from app.ui.widgets.media_pool import is_media, is_video
from app.ui.widgets.video_player import VideoPlayer, bgr_to_pixmap
import cv2
import numpy as np


def find_test_video() -> Path:
    folder = Path(r"D:\Dowland\One")
    if folder.is_dir():
        videos = [p for p in folder.iterdir() if is_video(p)]
        if videos:
            return videos[0]
    # fallback synthetic
    out = Path(tempfile.gettempdir()) / "sf_test_fight.mp4"
    w, h, fps = 640, 360, 15
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(fps * 3):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        cv2.rectangle(frame, (50 + i, 100), (120 + i, 220), (0, 0, 200), -1)
        writer.write(frame)
    writer.release()
    return out


def main() -> None:
    app = QApplication(sys.argv)

    video = find_test_video()
    print("VIDEO", video)
    assert video.exists(), "video missing"

    # 1) pixmap conversion (the bug that broke preview)
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None, "cannot read frame"
    pix = bgr_to_pixmap(frame)
    assert not pix.isNull(), "pixmap is null"
    assert pix.width() > 0 and pix.height() > 0
    print("PIXMAP_OK", pix.width(), pix.height())

    # 2) VideoPlayer.open + first frame on canvas
    player = VideoPlayer()
    player.resize(800, 450)
    player.show()
    app.processEvents()
    player.open(str(video))
    app.processEvents()
    assert player.duration > 0, f"duration is 0: {player.duration}"
    assert player.canvas.pixmap() is not None and not player.canvas.pixmap().isNull(), "canvas empty"
    print("PLAYER_OK", "duration", round(player.duration, 2), "size", player.frame_size)

    # 3) Media folder scan
    folder = video.parent
    media = [p for p in folder.iterdir() if is_media(p)]
    assert any(p.resolve() == video.resolve() for p in media), "video not listed as media"
    print("POOL_OK", len(media), "items")

    # 4) Project create + assign source path
    db_path = Path(tempfile.gettempdir()) / "sf_preview_test.db"
    if db_path.exists():
        db_path.unlink()
    db = Database(db_path)
    pid = db.create_project("Preview Test", media_folder=str(folder))
    info = probe_video(video)
    db.update_project(
        pid,
        source_path=str(video),
        duration=info["duration"],
        fps=info["fps"],
        width=int(info["width"]),
        height=int(info["height"]),
    )
    proj = db.get_project(pid)
    assert proj and Path(proj["source_path"]).exists()
    print("PROJECT_OK", pid)

    # 5) Short export clip from real source (first 2s, no tracking)
    out = Path(tempfile.gettempdir()) / "sf_preview_export.mp4"
    if out.exists():
        out.unlink()
    points = [
        TrackPoint(t=0.0, x=400, y=100, w=200, h=400, conf=1.0),
        TrackPoint(t=2.0, x=450, y=120, w=200, h=400, conf=1.0),
    ]
    render_clip(
        video,
        out,
        start_sec=1.0,
        end_sec=3.0,
        preset="reels",
        tracking_points=points,
        frame_w=int(info["width"]),
        frame_h=int(info["height"]),
        title="Aghayev",
        result="Test",
    )
    assert out.exists() and out.stat().st_size > 10_000, "export too small"
    print("EXPORT_OK", out.stat().st_size)

    # 6) Highlights on a short window — use proxy-like subsample via detecting on file
    # Keep it light: just ensure function returns list without crash
    cache = Path(tempfile.gettempdir()) / "sf_hl_cache"
    cache.mkdir(exist_ok=True)
    # For speed, skip full detect on 7min video in CI-like run — probe only first part by temp cut?
    # Fast path: call detect but it may take long. Instead verify import path works.
    from app.core.highlights import _motion_scores

    times, scores = _motion_scores(video, sample_fps=1.0, progress=None)
    # Only analyze by reading — for long video this still walks all frames at 1fps step
    # Actually _motion_scores reads every frame with step - for 10368 frames at step~24 it's ~400 samples - OK
    assert times.size > 0 and scores.size > 0
    print("MOTION_OK", times.size, float(scores.max()))

    player.close()
    app.quit()
    print("ALL_TESTS_PASSED")


if __name__ == "__main__":
    main()
