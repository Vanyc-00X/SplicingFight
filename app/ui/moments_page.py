from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QMessageBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
)

from app.core.crop import PRESETS
from app.core.highlights import detect_highlights
from app.core.tracking import (
    AthleteTracker,
    save_tracking,
    load_tracking,
    TrackPoint,
    tracking_quality,
)
from app.db.database import Database
from app.paths import CACHE_DIR, PROJECTS_DIR
from app.ui.widgets.video_player import VideoPlayer


class TrackWorker(QThread):
    progress = Signal(float, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        video: str,
        box: tuple[float, float, float, float],
        out_json: str,
        *,
        start_sec: float = 0.0,
        sample_fps: float = 12.0,
        max_seconds: float | None = None,
        target_color: str | None = None,
    ) -> None:
        super().__init__()
        self.video = video
        self.box = box
        self.out_json = out_json
        self.start_sec = start_sec
        self.sample_fps = sample_fps
        self.max_seconds = max_seconds
        self.target_color = target_color

    def run(self) -> None:
        try:
            tracker = AthleteTracker(model_name="yolov8n.pt", target_color=self.target_color)
            points = tracker.track_video(
                self.video,
                self.box,
                start_sec=self.start_sec,
                sample_fps=self.sample_fps,
                max_seconds=self.max_seconds,
                detect_every=3,
                target_color=self.target_color,
                progress=lambda p, m: self.progress.emit(p, m),
            )
            if len(points) < 2:
                raise RuntimeError("Трекинг не дал траекторию — кликните по бойцу ещё раз")
            # Require real motion in the path (frozen box = bad seed / wrong start)
            xs = [p.x + p.w * 0.5 for p in points]
            ys = [p.y + p.h * 0.5 for p in points]
            span = max(
                (max(xs) - min(xs)) if xs else 0.0,
                (max(ys) - min(ys)) if ys else 0.0,
            )
            if span < 6.0 and (points[-1].t - points[0].t) > 2.0:
                raise RuntimeError(
                    "Рамка почти не двигалась. Очистите трекинг, кликните по бойцу целиком и запустите снова."
                )
            save_tracking(self.out_json, points)
            self.finished_ok.emit(self.out_json)
        except Exception as exc:
            self.failed.emit(str(exc))


class HighlightWorker(QThread):
    progress = Signal(float, str)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        video: str,
        cache_dir: str,
        *,
        max_seconds: float | None = None,
    ) -> None:
        super().__init__()
        self.video = video
        self.cache_dir = cache_dir
        self.max_seconds = max_seconds

    def run(self) -> None:
        try:
            from app.core.device import device_label

            self.progress.emit(0.01, f"AI-моменты · {device_label()}")
            items = detect_highlights(
                self.video,
                cache_dir=self.cache_dir,
                max_seconds=self.max_seconds,
                use_pose=True,
                progress=lambda p, m: self.progress.emit(p, m),
            )
            self.finished_ok.emit(
                [
                    {
                        "start_sec": c.start_sec,
                        "end_sec": c.end_sec,
                        "score": c.score,
                        "kind": c.kind,
                        "thumbnail_path": c.thumbnail_path,
                        "accepted": 0,
                    }
                    for c in items
                ]
            )
        except Exception as exc:
            self.failed.emit(str(exc))


KIND_RU = {
    "action": "действие",
    "strike": "удар",
    "contact": "контакт",
    "crowd_reaction": "реакция зала",
    "manual": "вручную",
}


def resolve_project_video(project: dict | None) -> str | None:
    if not project:
        return None
    for key in ("proxy_path", "source_path"):
        path = project.get(key)
        if path and Path(path).exists():
            return str(path)
    folder = project.get("media_folder")
    if folder and Path(folder).is_dir():
        from app.ui.widgets.media_pool import is_video

        videos = sorted(
            [p for p in Path(folder).iterdir() if is_video(p)],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if videos:
            return str(videos[0])
    return None


class MomentsPage(QWidget):
    status = Signal(str)
    highlights_changed = Signal()

    def __init__(self, db: Database, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self.project_id: int | None = None
        self.project: dict | None = None
        self._video_path: str | None = None
        self._track_worker: TrackWorker | None = None
        self._hl_worker: HighlightWorker | None = None
        self._pending_box: tuple[float, float, float, float] | None = None
        self._pending_start: float = 0.0
        self._track_token = 0
        self._hl_token = 0

        self.player = VideoPlayer()
        self.player.athlete_selected.connect(self._on_athlete_click)

        self.btn_select = QPushButton("1. Кликнуть по спортсмену")
        self.btn_select.setCheckable(True)
        self.btn_track = QPushButton("2. Запустить трекинг")
        self.btn_track.setObjectName("Primary")
        self.btn_detect = QPushButton("3. AI: найти моменты")
        self.btn_detect.setObjectName("Primary")
        self.btn_accept = QPushButton("Принять")
        self.btn_accept.setObjectName("Success")
        self.btn_reject = QPushButton("Удалить")
        self.btn_add = QPushButton("Добавить вручную")
        self.btn_apply_bounds = QPushButton("Применить границы")
        self.btn_reload = QPushButton("Перезагрузить видео")

        self.btn_clear_track = QPushButton("Очистить трекинг")
        self.btn_clear_moments = QPushButton("Очистить моменты")
        self.btn_clear_all = QPushButton("Очистить всё")
        self.btn_redo_track = QPushButton("Заново: трекинг")
        self.btn_redo_track.setObjectName("Primary")
        self.btn_redo_moments = QPushButton("Заново: моменты")
        self.btn_redo_moments.setObjectName("Primary")

        self.cmb_preset = QComboBox()
        for key, cfg in PRESETS.items():
            self.cmb_preset.addItem(cfg["label"], key)
        self.cmb_preset.currentIndexChanged.connect(self._update_crop_preview)

        self.spin_start = QDoubleSpinBox()
        self.spin_end = QDoubleSpinBox()
        for spin in (self.spin_start, self.spin_end):
            spin.setRange(0, 99999)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)

        self.list_hl = QListWidget()
        self.list_hl.currentItemChanged.connect(self._on_hl_selected)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.lbl_help = QLabel(
            "Клик по бойцу на чётком кадре. Трекинг: ByteTrack + optical flow + цвет пояса/кимоно."
        )
        self.lbl_help.setObjectName("Subtitle")
        self.lbl_help.setWordWrap(True)
        self.lbl_video = QLabel("Видео не загружено")
        self.lbl_video.setObjectName("Subtitle")
        self.lbl_video.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Превью кропа", self.cmb_preset)
        form.addRow("Начало, c", self.spin_start)
        form.addRow("Конец, c", self.spin_end)

        tools = QVBoxLayout()
        tools.addWidget(self.lbl_video)
        tools.addWidget(self.btn_reload)
        tools.addWidget(self.btn_select)
        tools.addWidget(self.btn_track)
        tools.addWidget(self.btn_detect)

        reset_box = QGroupBox("Сброс / заново")
        reset_l = QVBoxLayout(reset_box)
        reset_row1 = QHBoxLayout()
        reset_row1.addWidget(self.btn_clear_track)
        reset_row1.addWidget(self.btn_clear_moments)
        reset_l.addLayout(reset_row1)
        reset_l.addWidget(self.btn_clear_all)
        reset_row2 = QHBoxLayout()
        reset_row2.addWidget(self.btn_redo_track)
        reset_row2.addWidget(self.btn_redo_moments)
        reset_l.addLayout(reset_row2)
        tools.addWidget(reset_box)

        tools.addWidget(QLabel("Моменты"))
        tools.addWidget(self.list_hl, 1)
        box = QGroupBox("Правка")
        box_l = QVBoxLayout(box)
        box_l.addLayout(form)
        row = QHBoxLayout()
        row.addWidget(self.btn_accept)
        row.addWidget(self.btn_reject)
        box_l.addLayout(row)
        box_l.addWidget(self.btn_apply_bounds)
        box_l.addWidget(self.btn_add)
        tools.addWidget(box)
        tools.addWidget(self.progress)
        tools.addWidget(self.lbl_help)

        root = QHBoxLayout(self)
        root.addWidget(self.player, 3)
        side = QWidget()
        side.setLayout(tools)
        side.setMaximumWidth(380)
        root.addWidget(side)

        self.btn_select.toggled.connect(self.player.set_select_mode)
        self.btn_track.clicked.connect(self.start_tracking)
        self.btn_detect.clicked.connect(self.start_detect)
        self.btn_accept.clicked.connect(lambda: self._set_accepted(True))
        self.btn_reject.clicked.connect(self._delete_current)
        self.btn_add.clicked.connect(self._add_manual)
        self.btn_apply_bounds.clicked.connect(self._apply_bounds)
        self.btn_reload.clicked.connect(self.reload_video)
        # clicked(bool) must not bind into keyword-only clear_* args
        self.btn_clear_track.clicked.connect(lambda: self.clear_tracking())
        self.btn_clear_moments.clicked.connect(lambda: self.clear_moments())
        self.btn_clear_all.clicked.connect(self.clear_all)
        self.btn_redo_track.clicked.connect(self.redo_tracking)
        self.btn_redo_moments.clicked.connect(self.redo_moments)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Always refresh when user opens this tab
        if self.project_id:
            self.load_project(self.project_id)

    def load_project(self, project_id: int) -> None:
        self.project_id = project_id
        self.project = self.db.get_project(project_id)
        if not self.project:
            self.lbl_video.setText("Проект не найден")
            return
        self.reload_video()
        points = load_tracking(self.project.get("tracking_json"))
        self.player.set_tracking(points)
        self._update_crop_preview()
        self.refresh_highlights()

    def reload_video(self) -> None:
        if self.project_id:
            self.project = self.db.get_project(self.project_id)
        video = resolve_project_video(self.project)
        self._video_path = video
        if not video:
            self.lbl_video.setText(
                "Нет видео. Media → выберите ролик → «Назначить исходником»."
            )
            self.lbl_help.setText("Cut пустой, пока не назначен исходник проекта.")
            self.status.emit("Cut: исходник не назначен")
            return
        try:
            self.player.open(video)
            # If source_path empty but we resolved from folder — persist it
            if self.project_id and self.project and not self.project.get("source_path"):
                from app.core.ffmpeg_utils import probe_video

                info = probe_video(video)
                self.db.update_project(
                    self.project_id,
                    source_path=video,
                    duration=info.get("duration", 0),
                    fps=info.get("fps", 25),
                    width=int(info.get("width") or 0),
                    height=int(info.get("height") or 0),
                    media_folder=str(Path(video).parent),
                )
                self.project = self.db.get_project(self.project_id)
            self.lbl_video.setText(
                f"{Path(video).name}\n"
                f"{self.player.frame_size[0]}x{self.player.frame_size[1]} · "
                f"{self.player.duration:.1f}s"
            )
            self.lbl_help.setText(
                "1) Включите «Кликнуть по спортсмену» и кликните по бойцу. "
                "2) Трекинг. 3) AI-моменты."
            )
            self.status.emit(f"Cut загружен: {Path(video).name}")
        except Exception as exc:
            self.lbl_video.setText(f"Ошибка загрузки: {exc}")
            QMessageBox.warning(self, "Cut / Моменты", str(exc))

    def _update_crop_preview(self) -> None:
        key = self.cmb_preset.currentData()
        aspect = PRESETS.get(key, PRESETS["reels"])["aspect"]
        self.player.set_crop_aspect(aspect)

    def _ensure_video(self) -> str | None:
        if self._video_path and Path(self._video_path).exists():
            return self._video_path
        self.reload_video()
        return self._video_path

    def _on_athlete_click(self, x: float, y: float, *_):
        video = self._ensure_video()
        if not video:
            QMessageBox.warning(self, "Трекинг", "Сначала загрузите видео (кнопка Перезагрузить)")
            return
        import cv2

        cap = cv2.VideoCapture(str(video))
        frame_idx = int(self.player.current_time() * max(self.player._fps, 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            QMessageBox.warning(self, "Трекинг", "Не удалось прочитать кадр")
            return
        tracker = AthleteTracker()
        boxes = tracker.detect_people(frame)
        chosen = tracker.pick_box(boxes, (x, y))
        if chosen is None:
            QMessageBox.information(
                self,
                "Трекинг",
                "Человек не найден. Остановите на кадре, где боец виден целиком, и кликните ещё раз.",
            )
            return
        self._pending_box = chosen[:4]
        t = self.player.current_time()
        self._pending_start = t
        self.player.set_tracking(
            [TrackPoint(t=t, x=chosen[0], y=chosen[1], w=chosen[2], h=chosen[3], conf=chosen[4])]
        )
        self.btn_select.setChecked(False)
        self.status.emit(f"Спортсмен выбран @ {t:.1f}s — запускайте трекинг")
        self.lbl_help.setText(
            f"Спортсмен выбран на {t:.1f}s. Нажмите «Запустить трекинг» — путь пойдёт от этого кадра."
        )

    def start_tracking(self) -> None:
        if not self.project_id:
            QMessageBox.warning(self, "Проект", "Сначала откройте проект на вкладке Media")
            return
        video = self._ensure_video()
        if not video:
            return
        if not self._pending_box:
            QMessageBox.warning(self, "Трекинг", "Сначала кликните по спортсмену (шаг 1)")
            return
        out = PROJECTS_DIR / f"project_{self.project_id}" / "tracking.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        self.progress.setVisible(True)
        self.btn_track.setEnabled(False)
        self._set_reset_enabled(False)
        color = None
        if self.project:
            color = self.project.get("target_athlete")  # red / blue
        self._invalidate_track_worker()
        self._track_token += 1
        token = self._track_token
        self._track_worker = TrackWorker(
            video,
            self._pending_box,
            str(out),
            start_sec=self._pending_start,
            sample_fps=12.0,
            target_color=color,
        )
        self._track_worker.progress.connect(self._on_progress)
        self._track_worker.finished_ok.connect(
            lambda path, t=token: self._on_track_done(path, t)
        )
        self._track_worker.failed.connect(lambda err, t=token: self._on_fail(err, t, "track"))
        self._track_worker.start()

    def _on_track_done(self, path: str, token: int) -> None:
        if token != self._track_token:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
            return
        self.btn_track.setEnabled(True)
        self._set_reset_enabled(True)
        self.progress.setVisible(False)
        self.db.update_project(self.project_id, tracking_json=path)
        self.project = self.db.get_project(self.project_id)
        points = load_tracking(path)
        self.player.set_tracking(points)
        q = tracking_quality(points)
        t0 = points[0].t if points else 0.0
        t1 = points[-1].t if points else 0.0
        self.status.emit(
            f"Трекинг {t0:.1f}–{t1:.1f}s · {len(points)} точек · покрытие {q['coverage']*100:.0f}%"
        )
        self.lbl_help.setText(
            f"Трекинг готов ({t0:.1f}–{t1:.1f}s, покрытие {q['coverage']*100:.0f}%). "
            "Прокрутите таймлайн — оранжевая рамка должна ехать за бойцом."
        )
        # Jump to click time so user immediately sees a moving track
        if points:
            self.player.seek(self._pending_start if self._pending_start <= t1 else t0)

    def start_detect(self) -> None:
        if not self.project_id:
            return
        video = self._ensure_video()
        if not video:
            return
        cache = CACHE_DIR / f"project_{self.project_id}"
        self.progress.setVisible(True)
        self.btn_detect.setEnabled(False)
        self._set_reset_enabled(False)
        self._invalidate_hl_worker()
        self._hl_token += 1
        token = self._hl_token
        self._hl_worker = HighlightWorker(video, str(cache))
        self._hl_worker.progress.connect(self._on_progress)
        self._hl_worker.finished_ok.connect(
            lambda items, t=token: self._on_hl_done(items, t)
        )
        self._hl_worker.failed.connect(lambda err, t=token: self._on_fail(err, t, "hl"))
        self._hl_worker.start()

    def _on_hl_done(self, items: list, token: int) -> None:
        if token != self._hl_token:
            return
        self.btn_detect.setEnabled(True)
        self._set_reset_enabled(True)
        self.progress.setVisible(False)
        self.db.replace_highlights(self.project_id, items)
        self.refresh_highlights()
        self.highlights_changed.emit()
        self.status.emit(f"Кандидатов: {len(items)}")
        self.lbl_help.setText(f"Найдено {len(items)} кандидатов. Примите нужные и перейдите в Edit.")

    def _on_progress(self, value: float, msg: str) -> None:
        self.progress.setValue(int(value * 100))
        self.lbl_help.setText(msg)
        self.status.emit(msg)

    def _on_fail(self, err: str, token: int | None = None, kind: str = "") -> None:
        if kind == "track" and token is not None and token != self._track_token:
            return
        if kind == "hl" and token is not None and token != self._hl_token:
            return
        self.btn_track.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self._set_reset_enabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Ошибка", err)

    def _disconnect_worker(self, worker: QThread | None) -> None:
        if worker is None:
            return
        for sig in (getattr(worker, "progress", None), getattr(worker, "finished_ok", None), getattr(worker, "failed", None)):
            if sig is None:
                continue
            try:
                sig.disconnect()
            except Exception:
                pass

    def _invalidate_track_worker(self) -> None:
        self._disconnect_worker(self._track_worker)
        if self._track_worker is not None and self._track_worker.isRunning():
            self._track_worker.requestInterruption()
        self._track_worker = None

    def _invalidate_hl_worker(self) -> None:
        self._disconnect_worker(self._hl_worker)
        if self._hl_worker is not None and self._hl_worker.isRunning():
            self._hl_worker.requestInterruption()
        self._hl_worker = None

    def _set_reset_enabled(self, enabled: bool) -> None:
        for btn in (
            self.btn_clear_track,
            self.btn_clear_moments,
            self.btn_clear_all,
            self.btn_redo_track,
            self.btn_redo_moments,
        ):
            btn.setEnabled(enabled)

    def refresh_highlights(self) -> None:
        self.list_hl.clear()
        if not self.project_id:
            return
        for h in self.db.list_highlights(self.project_id):
            mark = "OK" if h.get("accepted") else "-"
            kind = KIND_RU.get(h.get("kind"), h.get("kind"))
            text = f"{mark} {h['start_sec']:.1f}-{h['end_sec']:.1f}s  {kind}  ({h['score']:.2f})"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, h["id"])
            self.list_hl.addItem(item)

    def _on_hl_selected(self, current: QListWidgetItem | None, _prev) -> None:
        if not current:
            return
        hid = int(current.data(Qt.UserRole))
        for h in self.db.list_highlights(self.project_id):
            if h["id"] == hid:
                self.spin_start.setValue(float(h["start_sec"]))
                self.spin_end.setValue(float(h["end_sec"]))
                self.player.seek(float(h["start_sec"]))
                break

    def _current_hl_id(self) -> int | None:
        item = self.list_hl.currentItem()
        return int(item.data(Qt.UserRole)) if item else None

    def _set_accepted(self, accepted: bool) -> None:
        hid = self._current_hl_id()
        if hid is None:
            return
        self.db.update_highlight(hid, accepted=int(accepted))
        self.refresh_highlights()
        self.highlights_changed.emit()

    def _delete_current(self) -> None:
        hid = self._current_hl_id()
        if hid is None:
            return
        self.db.delete_highlight(hid)
        self.refresh_highlights()
        self.highlights_changed.emit()

    def _apply_bounds(self) -> None:
        hid = self._current_hl_id()
        if hid is None:
            return
        start = self.spin_start.value()
        end = self.spin_end.value()
        if end <= start:
            QMessageBox.warning(self, "Границы", "Конец должен быть больше начала")
            return
        self.db.update_highlight(hid, start_sec=start, end_sec=end)
        self.refresh_highlights()
        self.highlights_changed.emit()

    def _add_manual(self) -> None:
        if not self.project_id:
            return
        if not self._ensure_video():
            return
        t = self.player.current_time()
        self.db.add_highlight(
            self.project_id,
            start_sec=max(0, t - 1.5),
            end_sec=min(self.player.duration or t + 3.5, t + 3.5),
            score=1.0,
            kind="manual",
            accepted=1,
        )
        self.refresh_highlights()
        self.highlights_changed.emit()

    def clear_tracking(self, *, confirm: bool = True) -> bool:
        if not self.project_id:
            QMessageBox.information(self, "Сброс", "Сначала откройте проект")
            return False
        if confirm:
            ans = QMessageBox.question(
                self,
                "Очистить трекинг?",
                "Удалить трек спортсмена? Можно будет кликнуть и прогнать заново.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return False
        # Invalidate in-flight track so a late finish cannot restore the file/DB
        self._track_token += 1
        self._invalidate_track_worker()
        self.btn_track.setEnabled(True)
        self.progress.setVisible(False)
        self._set_reset_enabled(True)

        track_path = None
        if self.project:
            track_path = self.project.get("tracking_json")
        default_path = PROJECTS_DIR / f"project_{self.project_id}" / "tracking.json"
        self.db.update_project(self.project_id, tracking_json=None)
        for path in {track_path, str(default_path)}:
            if not path:
                continue
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
        self._pending_box = None
        self._pending_start = 0.0
        self.project = self.db.get_project(self.project_id)
        self.player.set_tracking([])
        self.btn_select.setChecked(False)
        self.player.set_select_mode(False)
        self.lbl_help.setText("Трекинг очищен. Кликните по спортсмену и запустите трекинг заново.")
        self.status.emit("Трекинг очищен")
        return True

    def clear_moments(self, *, confirm: bool = True) -> bool:
        if not self.project_id:
            QMessageBox.information(self, "Сброс", "Сначала откройте проект")
            return False
        if confirm:
            ans = QMessageBox.question(
                self,
                "Очистить моменты?",
                "Удалить все найденные/принятые моменты из списка и с таймлайна?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return False
        self._hl_token += 1
        self._invalidate_hl_worker()
        self.btn_detect.setEnabled(True)
        self.progress.setVisible(False)
        self._set_reset_enabled(True)

        self.db.clear_highlights(self.project_id)
        self.refresh_highlights()
        self.highlights_changed.emit()
        self.spin_start.setValue(0)
        self.spin_end.setValue(0)
        self.list_hl.clearSelection()
        self.lbl_help.setText("Моменты очищены. Можно снова запустить AI: найти моменты.")
        self.status.emit("Моменты очищены")
        return True

    def clear_all(self) -> None:
        if not self.project_id:
            QMessageBox.information(self, "Сброс", "Сначала откройте проект")
            return
        ans = QMessageBox.question(
            self,
            "Очистить всё?",
            "Удалить трекинг и все моменты. Видео останется.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self.clear_tracking(confirm=False)
        self.clear_moments(confirm=False)
        self.lbl_help.setText(
            "Всё очищено. 1) Клик по спортсмену → 2) трекинг → 3) AI-моменты."
        )
        self.status.emit("Cut: трекинг и моменты очищены")

    def redo_tracking(self) -> None:
        """Clear tracking and prepare for a fresh athlete click + track run."""
        if not self._ensure_video():
            QMessageBox.warning(self, "Трекинг", "Сначала загрузите видео")
            return
        if not self.clear_tracking(confirm=True):
            return
        self.btn_select.setChecked(True)
        self.player.set_select_mode(True)
        self.lbl_help.setText(
            "Заново: кликните по спортсмену на чётком кадре, затем «Запустить трекинг»."
        )
        self.status.emit("Готово к повторному трекингу — кликните по бойцу")

    def redo_moments(self) -> None:
        """Clear highlight list and immediately re-run AI detection."""
        if not self._ensure_video():
            QMessageBox.warning(self, "Моменты", "Сначала загрузите видео")
            return
        if not self.clear_moments(confirm=True):
            return
        self.lbl_help.setText("Повторный поиск интересных моментов…")
        self.start_detect()
