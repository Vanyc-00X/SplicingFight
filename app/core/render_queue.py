from __future__ import annotations

import json
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

import cv2

from app.core.export import concat_clips, render_clip
from app.core.tracking import load_tracking, scale_tracking
from app.db.database import Database
from app.paths import LIBRARY_DIR


class RenderWorker(QThread):
    job_updated = Signal(int, str, float, str)  # id, status, progress, message
    finished_all = Signal()

    def __init__(self, db: Database, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.db = db
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            job = self.db.next_queued_job()
            if not job:
                break
            self._process_job(job)
        self.finished_all.emit()

    def _process_job(self, job: dict) -> None:
        job_id = int(job["id"])
        project_id = int(job["project_id"])
        preset = job["preset"]
        payload = json.loads(job.get("payload_json") or "{}")
        self.db.update_render_job(job_id, status="running", progress=0, error=None)
        self.job_updated.emit(job_id, "running", 0.0, "Старт рендера")

        try:
            project = self.db.get_project(project_id)
            if not project:
                raise RuntimeError("Проект не найден")

            mode = payload.get("mode", "highlights")  # highlights | timeline | single
            out_dir = LIBRARY_DIR / f"project_{project_id}"
            out_dir.mkdir(parents=True, exist_ok=True)

            tracking_json = project.get("tracking_json")
            points = load_tracking(tracking_json)
            # Tracking usually runs on proxy — scale boxes to source resolution
            proxy = project.get("proxy_path")
            src_w = int(project.get("width") or 0)
            src_h = int(project.get("height") or 0)
            if proxy and points and src_w and src_h:
                cap = cv2.VideoCapture(str(proxy))
                pw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                ph = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                if pw and ph:
                    points = scale_tracking(points, src_w=pw, src_h=ph, dst_w=src_w, dst_h=src_h)

            def on_progress(p: float, msg: str) -> None:
                self.db.update_render_job(job_id, progress=p)
                self.job_updated.emit(job_id, "running", p, msg)

            outputs: list[Path] = []

            if mode == "single":
                start = float(payload["start_sec"])
                end = float(payload["end_sec"])
                out = out_dir / f"clip_{job_id}_{preset}.mp4"
                render_clip(
                    project["source_path"],
                    out,
                    start_sec=start,
                    end_sec=end,
                    preset=preset,
                    tracking_points=points,
                    frame_w=int(project["width"]),
                    frame_h=int(project["height"]),
                    title=project.get("title_text") or project.get("name"),
                    result=project.get("result_text"),
                    logo_path=project.get("logo_path"),
                    brand_color=project.get("brand_color") or "#C62828",
                    text_style=project.get("text_style") or "bold",
                    music_path=project.get("music_path"),
                    music_volume=float(project.get("music_volume") or 0.35),
                    muted=bool(payload.get("muted")),
                    progress=on_progress,
                )
                outputs = [out]
            elif mode == "timeline":
                clips = self.db.list_timeline(project_id)
                if not clips:
                    # fallback to accepted highlights
                    clips = [
                        {"start_sec": h["start_sec"], "end_sec": h["end_sec"], "muted": 0}
                        for h in self.db.list_highlights(project_id)
                        if h.get("accepted")
                    ]
                temp_paths: list[Path] = []
                for i, clip in enumerate(clips):
                    part = out_dir / f"part_{job_id}_{i}.mp4"
                    render_clip(
                        project["source_path"],
                        part,
                        start_sec=float(clip["start_sec"]),
                        end_sec=float(clip["end_sec"]),
                        preset=preset,
                        tracking_points=points,
                        frame_w=int(project["width"]),
                        frame_h=int(project["height"]),
                        title=project.get("title_text") or project.get("name"),
                        result=project.get("result_text"),
                        logo_path=project.get("logo_path"),
                        brand_color=project.get("brand_color") or "#C62828",
                        text_style=project.get("text_style") or "bold",
                        music_path=project.get("music_path") if i == 0 else None,
                        music_volume=float(project.get("music_volume") or 0.35),
                        muted=bool(clip.get("muted")),
                        progress=lambda p, m, i=i, n=len(clips): on_progress(
                            (i + p) / max(n, 1), m
                        ),
                    )
                    temp_paths.append(part)
                out = out_dir / f"timeline_{job_id}_{preset}.mp4"
                concat_clips(temp_paths, out, progress=on_progress)
                outputs = [out]
            else:
                highlights = [
                    h
                    for h in self.db.list_highlights(project_id)
                    if h.get("accepted") or payload.get("export_all")
                ]
                if payload.get("highlight_ids"):
                    ids = set(payload["highlight_ids"])
                    highlights = [h for h in highlights if h["id"] in ids]
                if not highlights:
                    raise RuntimeError("Нет выбранных моментов для экспорта")
                for i, h in enumerate(highlights):
                    out = out_dir / f"hl_{h['id']}_{preset}.mp4"
                    render_clip(
                        project["source_path"],
                        out,
                        start_sec=float(h["start_sec"]),
                        end_sec=float(h["end_sec"]),
                        preset=preset,
                        tracking_points=points,
                        frame_w=int(project["width"]),
                        frame_h=int(project["height"]),
                        title=project.get("title_text") or project.get("name"),
                        result=project.get("result_text"),
                        logo_path=project.get("logo_path"),
                        brand_color=project.get("brand_color") or "#C62828",
                        text_style=project.get("text_style") or "bold",
                        music_path=project.get("music_path"),
                        music_volume=float(project.get("music_volume") or 0.35),
                        progress=lambda p, m, i=i, n=len(highlights): on_progress(
                            (i + p) / max(n, 1), m
                        ),
                    )
                    outputs.append(out)
                    athlete = project.get("fighter_red") or project.get("fighter_blue") or ""
                    self.db.add_library_clip(
                        project_id=project_id,
                        title=f"{project.get('name')} · {h['kind']} {h['start_sec']:.1f}s",
                        athlete=athlete,
                        event_date=project.get("event_date"),
                        tags=h.get("kind", ""),
                        format=preset,
                        output_path=str(out),
                    )

            # register timeline/single in library too
            if mode in {"timeline", "single"} and outputs:
                athlete = project.get("fighter_red") or project.get("fighter_blue") or ""
                self.db.add_library_clip(
                    project_id=project_id,
                    title=f"{project.get('name')} · {mode}",
                    athlete=athlete,
                    event_date=project.get("event_date"),
                    tags=mode,
                    format=preset,
                    output_path=str(outputs[0]),
                )

            out_path = str(outputs[0]) if outputs else None
            self.db.update_render_job(
                job_id, status="done", progress=1.0, output_path=out_path, error=None
            )
            self.job_updated.emit(job_id, "done", 1.0, out_path or "Готово")
        except Exception as exc:
            err = f"{exc}\n{traceback.format_exc()}"
            self.db.update_render_job(job_id, status="failed", error=str(exc))
            self.job_updated.emit(job_id, "failed", 0.0, str(exc))
