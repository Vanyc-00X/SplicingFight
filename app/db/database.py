from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.paths import DB_PATH, ensure_dirs


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | None = None) -> None:
        ensure_dirs()
        self.path = path or DB_PATH
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    event_date TEXT,
                    event_name TEXT,
                    fighter_red TEXT,
                    fighter_blue TEXT,
                    category TEXT,
                    target_athlete TEXT DEFAULT 'red',
                    brand_color TEXT DEFAULT '#C62828',
                    logo_path TEXT,
                    music_path TEXT,
                    music_volume REAL DEFAULT 0.35,
                    title_text TEXT,
                    result_text TEXT,
                    text_style TEXT DEFAULT 'bold',
                    proxy_path TEXT,
                    duration REAL DEFAULT 0,
                    fps REAL DEFAULT 25,
                    width INTEGER DEFAULT 0,
                    height INTEGER DEFAULT 0,
                    tracking_json TEXT,
                    settings_json TEXT DEFAULT '{}',
                    media_folder TEXT
                );

                CREATE TABLE IF NOT EXISTS highlights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    score REAL DEFAULT 0,
                    kind TEXT DEFAULT 'action',
                    accepted INTEGER DEFAULT 0,
                    thumbnail_path TEXT,
                    notes TEXT,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS timeline_clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    highlight_id INTEGER,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    muted INTEGER DEFAULT 0,
                    order_index INTEGER NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY(highlight_id) REFERENCES highlights(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS library_clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    title TEXT NOT NULL,
                    athlete TEXT,
                    event_date TEXT,
                    tags TEXT DEFAULT '',
                    format TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS render_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    preset TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0,
                    output_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT DEFAULT '{}',
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );
                """
            )
            # migrate older DBs
            cols = {r[1] for r in conn.execute("PRAGMA table_info(projects)").fetchall()}
            if "media_folder" not in cols:
                conn.execute("ALTER TABLE projects ADD COLUMN media_folder TEXT")

    def create_project(
        self,
        name: str,
        source_path: str = "",
        *,
        duration: float = 0,
        fps: float = 25,
        width: int = 0,
        height: int = 0,
        proxy_path: str | None = None,
        event_date: str | None = None,
        event_name: str | None = None,
        fighter_red: str | None = None,
        fighter_blue: str | None = None,
        category: str | None = None,
        media_folder: str | None = None,
    ) -> int:
        now = _utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO projects (
                    name, source_path, created_at, updated_at, event_date, event_name,
                    fighter_red, fighter_blue, category, duration, fps, width, height,
                    proxy_path, media_folder
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    source_path or "",
                    now,
                    now,
                    event_date,
                    event_name,
                    fighter_red,
                    fighter_blue,
                    category,
                    duration,
                    fps,
                    width,
                    height,
                    proxy_path,
                    media_folder,
                ),
            )
            return int(cur.lastrowid)

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_project(self, project_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_project(self, project_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _utc_now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [project_id]
        with self.connect() as conn:
            conn.execute(f"UPDATE projects SET {cols} WHERE id = ?", values)

    def delete_project(self, project_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    def replace_highlights(self, project_id: int, items: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM highlights WHERE project_id = ?", (project_id,))
            for item in items:
                conn.execute(
                    """
                    INSERT INTO highlights (
                        project_id, start_sec, end_sec, score, kind, accepted, thumbnail_path, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        item["start_sec"],
                        item["end_sec"],
                        item.get("score", 0),
                        item.get("kind", "action"),
                        int(item.get("accepted", 0)),
                        item.get("thumbnail_path"),
                        item.get("notes"),
                    ),
                )
        self.update_project(project_id)

    def list_highlights(self, project_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM highlights
                WHERE project_id = ?
                ORDER BY start_sec ASC
                """,
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_highlight(self, highlight_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [highlight_id]
        with self.connect() as conn:
            conn.execute(f"UPDATE highlights SET {cols} WHERE id = ?", values)

    def delete_highlight(self, highlight_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM timeline_clips WHERE highlight_id = ?", (highlight_id,)
            )
            conn.execute("DELETE FROM highlights WHERE id = ?", (highlight_id,))

    def clear_highlights(self, project_id: int) -> None:
        """Remove all moments and timeline clips for a project."""
        with self.connect() as conn:
            conn.execute("DELETE FROM timeline_clips WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM highlights WHERE project_id = ?", (project_id,))
        self.update_project(project_id)

    def add_highlight(self, project_id: int, **fields: Any) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO highlights (
                    project_id, start_sec, end_sec, score, kind, accepted, thumbnail_path, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    fields.get("start_sec", 0),
                    fields.get("end_sec", 5),
                    fields.get("score", 0),
                    fields.get("kind", "manual"),
                    int(fields.get("accepted", 1)),
                    fields.get("thumbnail_path"),
                    fields.get("notes"),
                ),
            )
            return int(cur.lastrowid)

    def replace_timeline(self, project_id: int, clips: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM timeline_clips WHERE project_id = ?", (project_id,))
            for idx, clip in enumerate(clips):
                conn.execute(
                    """
                    INSERT INTO timeline_clips (
                        project_id, highlight_id, start_sec, end_sec, muted, order_index
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        clip.get("highlight_id"),
                        clip["start_sec"],
                        clip["end_sec"],
                        int(clip.get("muted", 0)),
                        clip.get("order_index", idx),
                    ),
                )
        self.update_project(project_id)

    def list_timeline(self, project_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM timeline_clips
                WHERE project_id = ?
                ORDER BY order_index ASC
                """,
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_library_clip(self, **fields: Any) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO library_clips (
                    project_id, title, athlete, event_date, tags, format, output_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fields.get("project_id"),
                    fields["title"],
                    fields.get("athlete"),
                    fields.get("event_date"),
                    fields.get("tags", ""),
                    fields["format"],
                    fields["output_path"],
                    _utc_now(),
                ),
            )
            return int(cur.lastrowid)

    def list_library(
        self,
        *,
        query: str | None = None,
        athlete: str | None = None,
        fmt: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM library_clips WHERE 1=1"
        params: list[Any] = []
        if query:
            sql += " AND (title LIKE ? OR tags LIKE ? OR athlete LIKE ?)"
            q = f"%{query}%"
            params.extend([q, q, q])
        if athlete:
            sql += " AND athlete LIKE ?"
            params.append(f"%{athlete}%")
        if fmt:
            sql += " AND format = ?"
            params.append(fmt)
        sql += " ORDER BY created_at DESC"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def create_render_job(
        self, project_id: int, preset: str, payload: dict[str, Any] | None = None
    ) -> int:
        now = _utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO render_jobs (
                    project_id, preset, status, progress, created_at, updated_at, payload_json
                ) VALUES (?, ?, 'queued', 0, ?, ?, ?)
                """,
                (project_id, preset, now, now, json.dumps(payload or {})),
            )
            return int(cur.lastrowid)

    def update_render_job(self, job_id: int, **fields: Any) -> None:
        fields["updated_at"] = _utc_now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [job_id]
        with self.connect() as conn:
            conn.execute(f"UPDATE render_jobs SET {cols} WHERE id = ?", values)

    def list_render_jobs(self, project_id: int | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if project_id is None:
                rows = conn.execute(
                    "SELECT * FROM render_jobs ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM render_jobs
                    WHERE project_id = ?
                    ORDER BY created_at DESC
                    """,
                    (project_id,),
                ).fetchall()
        return [dict(r) for r in rows]

    def next_queued_job(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM render_jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None
