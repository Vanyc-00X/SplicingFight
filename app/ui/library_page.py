from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QFileDialog,
)

from app.core.crop import PRESETS
import cv2

from app.core.export import render_clip
from app.core.tracking import load_tracking, scale_tracking
from app.db.database import Database


class LibraryPage(QWidget):
    status = Signal(str)

    def __init__(self, db: Database, parent=None) -> None:
        super().__init__(parent)
        self.db = db

        self.ed_query = QLineEdit()
        self.ed_query.setPlaceholderText("Поиск: спортсмен, тег, название…")
        self.cmb_format = QComboBox()
        self.cmb_format.addItem("Все форматы", "")
        for key, cfg in PRESETS.items():
            self.cmb_format.addItem(cfg["label"], key)

        self.btn_search = QPushButton("Найти")
        self.btn_open = QPushButton("Открыть файл")
        self.btn_folder = QPushButton("Папка")
        self.btn_reexport = QPushButton("Повторный экспорт")
        self.btn_reexport.setObjectName("Primary")
        self.btn_refresh = QPushButton("Обновить")

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Название", "Спортсмен", "Дата", "Формат", "Путь"]
        )
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        top = QHBoxLayout()
        top.addWidget(self.ed_query, 1)
        top.addWidget(self.cmb_format)
        top.addWidget(self.btn_search)
        top.addWidget(self.btn_refresh)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_open)
        btns.addWidget(self.btn_folder)
        btns.addWidget(self.btn_reexport)
        btns.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Библиотека клипов"))
        layout.addLayout(top)
        layout.addWidget(self.table, 1)
        layout.addLayout(btns)
        layout.addWidget(
            QLabel("Локальный индекс готовых роликов. Повторный экспорт использует исходник проекта.")
        )

        self.btn_search.clicked.connect(self.refresh)
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_open.clicked.connect(self.open_selected)
        self.btn_folder.clicked.connect(self.open_folder)
        self.btn_reexport.clicked.connect(self.reexport_selected)
        self.refresh()

    def refresh(self) -> None:
        q = self.ed_query.text().strip() or None
        fmt = self.cmb_format.currentData() or None
        rows = self.db.list_library(query=q, fmt=fmt)
        self.table.setRowCount(0)
        for row in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            values = [
                str(row["id"]),
                row.get("title") or "",
                row.get("athlete") or "",
                row.get("event_date") or "",
                row.get("format") or "",
                row.get("output_path") or "",
            ]
            for c, val in enumerate(values):
                item = QTableWidgetItem(val)
                if c == 0:
                    item.setData(Qt.UserRole, row)
                self.table.setItem(r, c, item)
        self.status.emit(f"Клипов в библиотеке: {len(rows)}")

    def _selected_row(self) -> dict | None:
        items = self.table.selectedItems()
        if not items:
            return None
        return self.table.item(items[0].row(), 0).data(Qt.UserRole)

    def open_selected(self) -> None:
        row = self._selected_row()
        if not row:
            return
        path = Path(row["output_path"])
        if not path.exists():
            QMessageBox.warning(self, "Файл", "Файл не найден на диске")
            return
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def open_folder(self) -> None:
        row = self._selected_row()
        if not row:
            return
        path = Path(row["output_path"]).parent
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def reexport_selected(self) -> None:
        row = self._selected_row()
        if not row:
            return
        project_id = row.get("project_id")
        if not project_id:
            QMessageBox.warning(self, "Экспорт", "Проект для клипа не связан")
            return
        project = self.db.get_project(int(project_id))
        if not project or not Path(project["source_path"]).exists():
            QMessageBox.warning(self, "Экспорт", "Исходник проекта недоступен")
            return

        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Куда сохранить",
            row.get("output_path") or "clip.mp4",
            "MP4 (*.mp4)",
        )
        if not dest:
            return

        # Use mid of title parse fallback: re-render first accepted highlight or full short window
        highlights = [h for h in self.db.list_highlights(int(project_id)) if h.get("accepted")]
        if not highlights:
            highlights = self.db.list_highlights(int(project_id))
        if not highlights:
            start, end = 0.0, min(8.0, float(project.get("duration") or 8))
        else:
            # try match by path name hl_{id}
            start = float(highlights[0]["start_sec"])
            end = float(highlights[0]["end_sec"])
            out_name = Path(row["output_path"]).name
            for h in highlights:
                if f"hl_{h['id']}_" in out_name:
                    start, end = float(h["start_sec"]), float(h["end_sec"])
                    break

        try:
            points = load_tracking(project.get("tracking_json"))
            proxy = project.get("proxy_path")
            src_w, src_h = int(project["width"]), int(project["height"])
            if proxy and points and Path(proxy).exists():
                cap = cv2.VideoCapture(str(proxy))
                pw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                ph = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                if pw and ph:
                    points = scale_tracking(points, src_w=pw, src_h=ph, dst_w=src_w, dst_h=src_h)
            render_clip(
                project["source_path"],
                dest,
                start_sec=start,
                end_sec=end,
                preset=row.get("format") or "reels",
                tracking_points=points,
                frame_w=src_w,
                frame_h=src_h,
                title=project.get("title_text") or project.get("name"),
                result=project.get("result_text"),
                logo_path=project.get("logo_path"),
                brand_color=project.get("brand_color") or "#C62828",
                text_style=project.get("text_style") or "bold",
                music_path=project.get("music_path"),
                music_volume=float(project.get("music_volume") or 0.35),
            )
            self.db.add_library_clip(
                project_id=project_id,
                title=f"Reexport · {row.get('title')}",
                athlete=row.get("athlete"),
                event_date=row.get("event_date"),
                tags="reexport",
                format=row.get("format") or "reels",
                output_path=dest,
            )
            self.refresh()
            self.status.emit(f"Повторный экспорт: {dest}")
        except Exception as exc:
            QMessageBox.critical(self, "Экспорт", str(exc))
