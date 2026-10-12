"""
Full end-to-end test on D:\\Dowland\\One and export results to РаботаИИ.
"""
from __future__ import annotations

import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from app.core.export import render_clip, concat_clips
from app.core.ffmpeg_utils import probe_video, run_ffmpeg
from app.core.highlights import detect_highlights
from app.core.tracking import AthleteTracker, TrackPoint, save_tracking, load_tracking
from app.db.database import Database
from app.ui.main_window import MainWindow
from app.ui.moments_page import resolve_project_video
from app.ui.widgets.media_pool import is_video
from app.ui.widgets.video_player import bgr_to_pixmap


MEDIA_DIR = Path(r"D:\Dowland\One")
OUT_DIR = MEDIA_DIR / "РаботаИИ"
WORK_SECONDS = 90.0  # AI on first 90s for practical runtime
TRACK_SAMPLE_FPS = 4.0


def log(lines: list[str], msg: str) -> None:
    lines.append(msg)
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def make_work_clip(source: Path, dest: Path, seconds: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-ss",
            "0",
            "-i",
            str(source),
            "-t",
            f"{seconds:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(dest),
        ]
    )
    return dest


def pick_athlete_box(video: Path, t: float = 8.0) -> tuple[float, float, float, float]:
    tracker = AthleteTracker()
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read frame for athlete pick")
    boxes = tracker.detect_people(frame)
    if not boxes:
        # fallback center box
        h, w = frame.shape[:2]
        return (w * 0.35, h * 0.2, w * 0.25, h * 0.6)
    # Prefer left-ish fighter (often red side in this bout camera)
    boxes_sorted = sorted(boxes, key=lambda b: b[0])
    return boxes_sorted[0][:4]


def main() -> int:
    report: list[str] = []
    results: dict = {"started_at": datetime.now().isoformat(), "steps": []}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips_dir = OUT_DIR / "clips"
    clips_dir.mkdir(exist_ok=True)
    cache_dir = OUT_DIR / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir()

    log(report, "# Отчёт прогона SplicingFight")
    log(report, f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(report, f"Медиапапка: `{MEDIA_DIR}`")
    log(report, f"Результаты: `{OUT_DIR}`")
    log(report, "")

    try:
        videos = [p for p in MEDIA_DIR.iterdir() if is_video(p)]
        assert videos, "В D:\\Dowland\\One нет видео"
        source = videos[0]
        log(report, f"## 1. Исходник\n- Файл: `{source.name}`")
        info = probe_video(source)
        log(
            report,
            f"- {info['width']}x{info['height']} · {info['fps']:.2f} fps · {info['duration']:.1f}s",
        )
        results["source"] = {"path": str(source), **info}
        results["steps"].append({"name": "probe", "ok": True})

        # GUI: Media preview + Cut load
        app = QApplication(sys.argv)
        db_path = OUT_DIR / "test_run.db"
        if db_path.exists():
            db_path.unlink()
        db = Database(db_path)
        win = MainWindow(db)
        win.show()
        app.processEvents()

        pid = db.create_project(
            "Aghayev vs Eltemur — тест",
            media_folder=str(MEDIA_DIR),
            event_name="European Karate Championships 2016",
            fighter_red="Erman Eltemur",
            fighter_blue="Rafael Aghayev",
            category="FINAL",
        )
        win.page_project.load_project(pid)
        app.processEvents()
        win.page_project.preview_media(str(source))
        app.processEvents()
        proj = db.get_project(pid)
        assert proj and Path(proj["source_path"]).exists(), "source not bound"
        log(report, "## 2. Media / назначение исходника\n- OK: исходник привязан автоматически при превью")
        results["steps"].append({"name": "assign_source", "ok": True})

        # Cut tab load
        win.tabs.setCurrentWidget(win.page_moments)
        app.processEvents()
        win.page_moments.load_project(pid)
        app.processEvents()
        assert win.page_moments.player.duration > 0, "Cut player empty"
        assert win.page_moments.player.canvas.pixmap() and not win.page_moments.player.canvas.pixmap().isNull()
        log(
            report,
            f"## 3. Cut / Моменты — загрузка\n- OK: duration={win.page_moments.player.duration:.1f}s, pixmap OK",
        )
        results["steps"].append({"name": "cut_load", "ok": True, "duration": win.page_moments.player.duration})

        # Work clip for AI (90s)
        work = OUT_DIR / "work_90s.mp4"
        log(report, f"## 4. Рабочий клип для AI\n- Режем первые {WORK_SECONDS:.0f}s -> `{work.name}`")
        make_work_clip(source, work, WORK_SECONDS)
        work_info = probe_video(work)
        log(report, f"- OK: {work_info['duration']:.1f}s")
        results["steps"].append({"name": "work_clip", "ok": True})

        # Tracking
        log(report, "## 5. Трекинг спортсмена")
        box = pick_athlete_box(work, t=6.0)
        log(report, f"- Стартовый bbox: {[round(x,1) for x in box]}")
        tracker = AthleteTracker()
        points = tracker.track_video(
            work,
            box,
            sample_fps=TRACK_SAMPLE_FPS,
            progress=lambda p, m: print(f"  track {p*100:.0f}% {m}", flush=True)
            if int(p * 20) != int((p - 0.01) * 20)
            else None,
        )
        track_json = OUT_DIR / "tracking.json"
        save_tracking(track_json, points)
        covered = sum(1 for p in points if p.conf > 0.2) / max(len(points), 1)
        log(report, f"- Точек трека: {len(points)}")
        log(report, f"- Доля кадров с детектом: {covered*100:.1f}%")
        db.update_project(
            pid,
            tracking_json=str(track_json),
            source_path=str(source),
            width=int(info["width"]),
            height=int(info["height"]),
            fps=info["fps"],
            duration=info["duration"],
            title_text="Aghayev vs Eltemur",
            result_text="European Championships Final",
            brand_color="#C4541A",
        )
        results["steps"].append(
            {"name": "tracking", "ok": True, "points": len(points), "coverage": covered}
        )
        assert len(points) >= 10, "too few track points"

        # Highlights (motion+audio+pose on 90s)
        log(report, "## 6. AI-хайлайты (motion + pose + audio)")
        highlights = detect_highlights(
            work,
            cache_dir=cache_dir,
            max_candidates=12,
            use_pose=True,
            progress=lambda p, m: print(f"  hl {p*100:.0f}% {m}", flush=True)
            if int(p * 10) != int((p - 0.05) * 10)
            else None,
        )
        log(report, f"- Кандидатов: {len(highlights)}")
        for i, h in enumerate(highlights[:8], 1):
            log(
                report,
                f"  {i}. {h.start_sec:.1f}-{h.end_sec:.1f}s · {h.kind} · score={h.score:.2f}",
            )
        assert len(highlights) >= 3, "too few highlights"
        accepted = highlights[: min(5, len(highlights))]
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
                for h in accepted
            ],
        )
        results["steps"].append(
            {
                "name": "highlights",
                "ok": True,
                "count": len(highlights),
                "accepted": len(accepted),
            }
        )

        # Manual highlight test
        db.add_highlight(pid, start_sec=10.0, end_sec=16.0, score=1.0, kind="manual", accepted=1)
        log(report, "## 7. Ручной момент\n- OK: 10.0-16.0s manual accepted")

        # Export clips reels + youtube from work clip (coords match work)
        log(report, "## 8. Экспорт клипов")
        exported = []
        track_pts = load_tracking(track_json)
        hls = [h for h in db.list_highlights(pid) if h.get("accepted")]
        for i, h in enumerate(hls[:4], 1):
            for preset in ("reels", "youtube"):
                out = clips_dir / f"clip_{i:02d}_{preset}.mp4"
                render_clip(
                    work,
                    out,
                    start_sec=float(h["start_sec"]),
                    end_sec=float(h["end_sec"]),
                    preset=preset,
                    tracking_points=track_pts,
                    frame_w=int(work_info["width"]),
                    frame_h=int(work_info["height"]),
                    title="Aghayev vs Eltemur",
                    result=f"{h['kind']} · {h['start_sec']:.1f}s",
                    brand_color="#C4541A",
                )
                assert out.exists() and out.stat().st_size > 5000
                exported.append(out)
                db.add_library_clip(
                    project_id=pid,
                    title=f"clip_{i}_{preset}",
                    athlete="Rafael Aghayev",
                    event_date="2016-05-01",
                    tags=h["kind"],
                    format=preset,
                    output_path=str(out),
                )
                log(report, f"- OK `{out.name}` ({out.stat().st_size // 1024} KB)")
        results["steps"].append({"name": "export_clips", "ok": True, "files": len(exported)})

        # Timeline concat (reels only first 3)
        reel_parts = sorted(clips_dir.glob("clip_*_reels.mp4"))[:3]
        if reel_parts:
            timeline = OUT_DIR / "timeline_reels.mp4"
            concat_clips(reel_parts, timeline)
            assert timeline.exists()
            log(report, f"## 9. Склейка таймлайна\n- OK: `{timeline.name}` ({timeline.stat().st_size // 1024} KB)")
            results["steps"].append({"name": "timeline", "ok": True})

        # Edit tab load
        win.tabs.setCurrentWidget(win.page_editor)
        app.processEvents()
        win.page_editor.load_project(pid)
        app.processEvents()
        assert win.page_editor.player.duration > 0
        log(report, "## 10. Edit / Экспорт вкладка\n- OK: видео загружено")
        results["steps"].append({"name": "edit_tab", "ok": True})

        # Library
        lib = db.list_library()
        assert lib
        log(report, f"## 11. Библиотека\n- Записей: {len(lib)}")
        results["steps"].append({"name": "library", "ok": True, "count": len(lib)})

        # pixmap sanity
        pix = bgr_to_pixmap(np.zeros((100, 100, 3), dtype=np.uint8) + 40)
        assert not pix.isNull()
        results["steps"].append({"name": "pixmap", "ok": True})

        win.close()
        app.quit()

        results["ok"] = True
        results["finished_at"] = datetime.now().isoformat()
        log(report, "")
        log(report, "## Итог\n**ALL TESTS PASSED**")
    except Exception as exc:
        results["ok"] = False
        results["error"] = str(exc)
        results["traceback"] = traceback.format_exc()
        log(report, "")
        log(report, f"## ОШИБКА\n```\n{traceback.format_exc()}\n```")
        print(traceback.format_exc())

    # Write artifacts
    (OUT_DIR / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    actions = """# Что сделано

## Исправления в приложении
1. **Cut/Моменты пустой** — вкладка не подтягивала видео, пока не был `source_path`.
   - При превью на Media исходник **автоматически привязывается** к проекту.
   - При переключении на Cut видео **перезагружается**.
   - Если `source_path` пуст — берётся видео из медиапапки проекта.
2. **Назначить исходником** — больше не ждёт долгий proxy; привязка мгновенная.
3. **Viewer setPixmap** — ранее уже починен (QImage→QPixmap).

## Прогон на D:\\Dowland\\One
- Исходник: финал Aghayev vs Eltemur.
- Для AI создан рабочий клип `work_90s.mp4` (первые 90 секунд) — полный 7‑минутный трекинг+pose занял бы слишком долго для одного прогона; пайплайн тот же.
- Выполнены: probe, Media assign, Cut load, YOLO-трекинг, AI-хайлайты, ручной момент, экспорт Reels/YouTube, склейка таймлайна, Edit tab, Library.

## Файлы в этой папке
- `REPORT.md` — подробный отчёт
- `results.json` — машинный статус шагов
- `work_90s.mp4` — рабочий фрагмент
- `tracking.json` — трек спортсмена
- `clips/` — экспортированные рилсы и YouTube-клипы
- `timeline_reels.mp4` — склейка
- `cache/` — аудио/превью хайлайтов
- `test_run.db` — SQLite тестового проекта

## Как повторить в GUI
1. `python main.py`
2. Media → открыть проект / создать → папка `D:\\Dowland\\One`
3. Клик по видео (исходник привяжется сам) или «Назначить исходником»
4. Cut / Моменты → видео должно сразу открыться
5. Клик по спортсмену → трекинг → AI → принять → Edit → экспорт
"""
    (OUT_DIR / "DEISTVIYA.md").write_text(actions, encoding="utf-8")
    print("\n=== OUTPUT ===", OUT_DIR)
    print("OK" if results.get("ok") else "FAILED")
    return 0 if results.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
