from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize, QThread, Signal, QMutex
from PySide6.QtGui import QIcon, QImage, QPixmap, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QComboBox,
    QLineEdit,
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wmv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_media(path: Path) -> bool:
    return path.is_file() and (is_video(path) or is_image(path))


class ThumbWorker(QThread):
    thumb_ready = Signal(str, QPixmap)
    finished_scan = Signal(list)  # list[str paths]

    def __init__(self, folder: str, filter_mode: str = "all") -> None:
        super().__init__()
        self.folder = folder
        self.filter_mode = filter_mode
        self._stop = False
        self._mutex = QMutex()

    def stop(self) -> None:
        self._mutex.lock()
        self._stop = True
        self._mutex.unlock()

    def _stopped(self) -> bool:
        self._mutex.lock()
        v = self._stop
        self._mutex.unlock()
        return v

    def run(self) -> None:
        root = Path(self.folder)
        if not root.is_dir():
            self.finished_scan.emit([])
            return
        paths: list[Path] = []
        try:
            for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
                if self._stopped():
                    return
                if not is_media(p):
                    continue
                if self.filter_mode == "video" and not is_video(p):
                    continue
                if self.filter_mode == "photo" and not is_image(p):
                    continue
                paths.append(p)
        except OSError:
            self.finished_scan.emit([])
            return

        self.finished_scan.emit([str(p) for p in paths])
        for p in paths:
            if self._stopped():
                return
            pix = self._make_thumb(p)
            if pix is not None:
                self.thumb_ready.emit(str(p), pix)

    def _make_thumb(self, path: Path) -> QPixmap | None:
        try:
            if is_image(path):
                img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    return self._placeholder(path, "VIDEO")
                # mid frame if possible
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if frames > 10:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 10)
                ok, img = cap.read()
                cap.release()
                if not ok or img is None:
                    return self._placeholder(path, "VIDEO")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg).scaled(
                160, 90, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            # center crop to 160x90
            if pix.width() > 160 or pix.height() > 90:
                x = max((pix.width() - 160) // 2, 0)
                y = max((pix.height() - 90) // 2, 0)
                pix = pix.copy(x, y, min(160, pix.width()), min(90, pix.height()))
            return self._badge(pix, "VID" if is_video(path) else "IMG")
        except Exception:
            return self._placeholder(path, "VID" if is_video(path) else "IMG")

    def _placeholder(self, path: Path, kind: str) -> QPixmap:
        pix = QPixmap(160, 90)
        pix.fill(QColor("#1e2228"))
        return self._badge(pix, kind)

    def _badge(self, pix: QPixmap, kind: str) -> QPixmap:
        out = QPixmap(pix)
        painter = QPainter(out)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(0, 0, 36, 16, QColor(0, 0, 0, 160))
        painter.setPen(QColor("#f0a030") if kind.startswith("VID") else QColor("#6cb6ff"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(4, 12, kind[:3])
        painter.end()
        return out


class MediaPoolWidget(QWidget):
    """DaVinci-style media pool: open folder, show video/photo thumbs."""

    folder_changed = Signal(str)
    media_activated = Signal(str)  # path
    media_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._folder: str | None = None
        self._worker: ThumbWorker | None = None
        self._items: dict[str, QListWidgetItem] = {}

        self.lbl_title = QLabel("Media Pool")
        self.lbl_title.setObjectName("PanelTitle")
        self.lbl_path = QLabel("Папка не выбрана")
        self.lbl_path.setObjectName("Subtitle")
        self.lbl_path.setWordWrap(True)

        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("Фильтр по имени…")
        self.cmb_kind = QComboBox()
        self.cmb_kind.addItem("Все", "all")
        self.cmb_kind.addItem("Видео", "video")
        self.cmb_kind.addItem("Фото", "photo")

        self.btn_folder = QPushButton("Открыть папку")
        self.btn_folder.setObjectName("Primary")
        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setFixedWidth(36)

        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(160, 90))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setMovement(QListWidget.Static)
        self.list.setSpacing(8)
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(True)
        self.list.setObjectName("MediaPool")

        top = QHBoxLayout()
        top.addWidget(self.btn_folder, 1)
        top.addWidget(self.btn_refresh)

        filters = QHBoxLayout()
        filters.addWidget(self.cmb_kind)
        filters.addWidget(self.ed_filter, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_path)
        layout.addLayout(top)
        layout.addLayout(filters)
        layout.addWidget(self.list, 1)

        self.btn_folder.clicked.connect(self.choose_folder)
        self.btn_refresh.clicked.connect(self.refresh)
        self.cmb_kind.currentIndexChanged.connect(self.refresh)
        self.ed_filter.textChanged.connect(self._apply_name_filter)
        self.list.itemDoubleClicked.connect(self._on_activate)
        self.list.currentItemChanged.connect(self._on_select)

    @property
    def folder(self) -> str | None:
        return self._folder

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка с медиа", self._folder or "")
        if path:
            self.set_folder(path)

    def set_folder(self, folder: str) -> None:
        self._folder = folder
        self.lbl_path.setText(folder)
        self.folder_changed.emit(folder)
        self.refresh()

    def refresh(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(500)
        self.list.clear()
        self._items.clear()
        if not self._folder:
            return
        mode = self.cmb_kind.currentData() or "all"
        self._worker = ThumbWorker(self._folder, mode)
        self._worker.finished_scan.connect(self._on_scan)
        self._worker.thumb_ready.connect(self._on_thumb)
        self._worker.start()

    def _on_scan(self, paths: list[str]) -> None:
        query = self.ed_filter.text().strip().lower()
        for path in paths:
            name = Path(path).name
            if query and query not in name.lower():
                continue
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            item.setSizeHint(QSize(168, 120))
            # placeholder icon
            ph = QPixmap(160, 90)
            ph.fill(QColor("#1e2228"))
            item.setIcon(QIcon(ph))
            self.list.addItem(item)
            self._items[path] = item

    def _on_thumb(self, path: str, pix: QPixmap) -> None:
        item = self._items.get(path)
        if item:
            item.setIcon(QIcon(pix))

    def _apply_name_filter(self, _text: str) -> None:
        query = self.ed_filter.text().strip().lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            name = item.text().lower()
            item.setHidden(bool(query) and query not in name)

    def _on_activate(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            self.media_activated.emit(str(path))

    def _on_select(self, current: QListWidgetItem | None, _prev) -> None:
        if current:
            path = current.data(Qt.UserRole)
            if path:
                self.media_selected.emit(str(path))

    def selected_path(self) -> str | None:
        item = self.list.currentItem()
        if not item:
            return None
        return str(item.data(Qt.UserRole))
