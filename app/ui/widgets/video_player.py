from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QMouseEvent
from PySide6.QtWidgets import QLabel, QHBoxLayout, QPushButton, QSlider, QVBoxLayout, QWidget

from app.core.crop import crop_at_time
from app.core.tracking import TrackPoint, box_at_time


def bgr_to_pixmap(frame_bgr: np.ndarray, target_size=None) -> QPixmap:
    """Convert OpenCV BGR frame to QPixmap (optionally scaled)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return QPixmap()
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    if target_size is not None and target_size.width() >= 2 and target_size.height() >= 2:
        pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


class VideoCanvas(QLabel):
    clicked = Signal(float, float)  # x, y in source frame coords

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#0d0f12; border:1px solid #2a2a2a; border-radius:4px;")
        self._frame: np.ndarray | None = None
        self._scale = 1.0
        self._offset = (0, 0)
        self._box: tuple[float, float, float, float] | None = None
        self._crop = None
        self._select_mode = False

    def set_select_mode(self, enabled: bool) -> None:
        self._select_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def show_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        box: tuple[float, float, float, float] | None = None,
        crop=None,
    ) -> None:
        self._frame = frame_bgr
        self._box = box
        self._crop = crop
        self._paint()

    def clear_frame(self) -> None:
        self._frame = None
        self.clear()

    def _paint(self) -> None:
        if self._frame is None:
            return
        frame = self._frame.copy()
        if self._box is not None:
            x, y, w, h = [int(v) for v in self._box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 80, 255), 2)
        if self._crop is not None:
            cv2.rectangle(
                frame,
                (self._crop.x, self._crop.y),
                (self._crop.x + self._crop.w, self._crop.y + self._crop.h),
                (80, 220, 120),
                2,
            )
        fh, fw = frame.shape[:2]
        pix = bgr_to_pixmap(frame, self.size())
        if pix.isNull():
            return
        self._scale = pix.width() / fw if fw else 1.0
        ox = (self.width() - pix.width()) // 2
        oy = (self.height() - pix.height()) // 2
        self._offset = (ox, oy)
        self.setPixmap(pix)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._frame is not None:
            self._paint()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._select_mode or self._frame is None or self._scale <= 0:
            return
        ox, oy = self._offset
        x = (event.position().x() - ox) / self._scale
        y = (event.position().y() - oy) / self._scale
        fh, fw = self._frame.shape[:2]
        if 0 <= x < fw and 0 <= y < fh:
            self.clicked.emit(float(x), float(y))


class VideoPlayer(QWidget):
    position_changed = Signal(float)
    athlete_selected = Signal(float, float, float, float)  # x,y,w,h
    opened = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.canvas = VideoCanvas()
        self.canvas.clicked.connect(self._on_click)

        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedWidth(52)
        self.slider = QSlider(Qt.Horizontal)
        self.lbl_time = QLabel("00:00 / 00:00")

        controls = QHBoxLayout()
        controls.addWidget(self.btn_play)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.lbl_time)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(controls)

        self._cap: cv2.VideoCapture | None = None
        self._path: str | None = None
        self._fps = 25.0
        self._duration = 0.0
        self._playing = False
        self._frame_w = 0
        self._frame_h = 0
        self._tracking: list[TrackPoint] = []
        self._crop_aspect: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

        self.btn_play.clicked.connect(self.toggle_play)
        self.slider.sliderPressed.connect(self._pause_for_seek)
        self.slider.sliderReleased.connect(self._seek_from_slider)
        # Live scrub — otherwise box looks "frozen" while dragging the timeline
        self.slider.valueChanged.connect(self._scrub_while_dragging)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def path(self) -> str | None:
        return self._path

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_w, self._frame_h

    def open(self, path: str | Path) -> None:
        path = str(path)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._playing = False
        self.btn_play.setText("Play")
        self._timer.stop()

        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {path}")

        self._cap = cap
        self._path = path
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        self._frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._duration = frames / self._fps if self._fps and frames > 0 else 0.0

        # Fallback duration via probe if OpenCV reports 0
        if self._duration <= 0:
            try:
                from app.core.ffmpeg_utils import probe_video

                info = probe_video(path)
                self._duration = float(info.get("duration") or 0)
                if info.get("fps"):
                    self._fps = float(info["fps"])
                if info.get("width"):
                    self._frame_w = int(info["width"])
                if info.get("height"):
                    self._frame_h = int(info["height"])
            except Exception:
                pass

        self.slider.setRange(0, max(int(self._duration * 1000), 1))
        # Read first frame without relying on seek (more reliable)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Видео открыто, но кадр не читается: {path}")
        self._show(frame, 0.0)
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.opened.emit(path)

    def set_tracking(self, points: list[TrackPoint]) -> None:
        self._tracking = points
        self._refresh_current()

    def set_crop_aspect(self, aspect: float | None) -> None:
        self._crop_aspect = aspect
        self._refresh_current()

    def set_select_mode(self, enabled: bool) -> None:
        self.canvas.set_select_mode(enabled)

    def current_time(self) -> float:
        return self.slider.value() / 1000.0

    def seek(self, t: float) -> None:
        if not self._cap:
            return
        t = max(0.0, min(t, self._duration if self._duration > 0 else t))
        # Prefer frame index seek — more reliable than POS_MSEC on some mp4
        frame_idx = int(t * self._fps) if self._fps > 0 else 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = self._cap.read()
        if ok:
            self._show(frame, t)
            self.slider.blockSignals(True)
            self.slider.setValue(int(t * 1000))
            self.slider.blockSignals(False)
            self.position_changed.emit(t)

    def toggle_play(self) -> None:
        if not self._cap:
            return
        self._playing = not self._playing
        self.btn_play.setText("Pause" if self._playing else "Play")
        if self._playing:
            self._timer.start()
        else:
            self._timer.stop()

    def _pause_for_seek(self) -> None:
        self._timer.stop()

    def _scrub_while_dragging(self, value: int) -> None:
        if self.slider.isSliderDown():
            self.seek(value / 1000.0)

    def _seek_from_slider(self) -> None:
        self.seek(self.slider.value() / 1000.0)
        if self._playing:
            self._timer.start()

    def _tick(self) -> None:
        if not self._cap:
            return
        ok, frame = self._cap.read()
        if not ok:
            self._playing = False
            self.btn_play.setText("Play")
            self._timer.stop()
            return
        pos_frames = float(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        t = (pos_frames - 1) / self._fps if self._fps else 0.0
        self._show(frame, max(t, 0.0))
        self.slider.blockSignals(True)
        self.slider.setValue(int(max(t, 0.0) * 1000))
        self.slider.blockSignals(False)
        self.position_changed.emit(t)

    def _refresh_current(self) -> None:
        if self._cap:
            self.seek(self.current_time())

    def _show(self, frame: np.ndarray, t: float) -> None:
        box = box_at_time(self._tracking, t)
        crop = None
        if self._crop_aspect and self._frame_w and self._frame_h:
            crop = crop_at_time(self._tracking, t, self._frame_w, self._frame_h, self._crop_aspect)
        self.canvas.show_frame(frame, box=box, crop=crop)
        self.lbl_time.setText(f"{self._fmt(t)} / {self._fmt(self._duration)}")

    def _on_click(self, x: float, y: float) -> None:
        if not self._cap:
            return
        self.athlete_selected.emit(x, y, 0, 0)

    @staticmethod
    def _fmt(t: float) -> str:
        m = int(t // 60)
        s = int(t % 60)
        return f"{m:02d}:{s:02d}"

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        super().closeEvent(event)
