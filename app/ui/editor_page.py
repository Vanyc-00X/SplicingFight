from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QLineEdit,
    QFormLayout,
    QGroupBox,
    QComboBox,
    QDoubleSpinBox,
    QMessageBox,
    QProgressBar,
    QCheckBox,
    QColorDialog,
)

from app.core.crop import PRESETS
from app.core.render_queue import RenderWorker
from app.db.database import Database
from app.ui.widgets.video_player import VideoPlayer


class EditorPage(QWidget):
    status = Signal(str)

    def __init__(self, db: Database, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self.project_id: int | None = None
        self.project: dict | None = None
        self._worker: RenderWorker | None = None

        self.player = VideoPlayer()

        self.list_timeline = QListWidget()
        self.list_jobs = QListWidget()

        self.ed_title = QLineEdit()
        self.ed_result = QLineEdit()
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(["bold", "light"])
        self.ed_logo = QLineEdit()
        self.ed_music = QLineEdit()
        self.spin_music = QDoubleSpinBox()
        self.spin_music.setRange(0, 1)
        self.spin_music.setSingleStep(0.05)
        self.spin_music.setValue(0.35)
        self.ed_brand = QLineEdit("#C62828")

        self.cmb_preset = QComboBox()
        for key, cfg in PRESETS.items():
            self.cmb_preset.addItem(cfg["label"], key)

        self.chk_export_all = QCheckBox("Экспорт всех кандидатов (не только принятых)")

        form = QFormLayout()
        form.addRow("Заголовок", self.ed_title)
        form.addRow("Результат", self.ed_result)
        form.addRow("Стиль текста", self.cmb_style)
        form.addRow("Цвет бренда", self.ed_brand)
        form.addRow("Логотип", self.ed_logo)
        form.addRow("Музыка", self.ed_music)
        form.addRow("Громкость музыки", self.spin_music)
        form.addRow("Пресет экспорта", self.cmb_preset)

        self.btn_logo = QPushButton("Выбрать лого…")
        self.btn_music = QPushButton("Выбрать трек…")
        self.btn_color = QPushButton("Цвет…")
        self.btn_save = QPushButton("Сохранить монтаж")
        self.btn_build_timeline = QPushButton("Собрать таймлайн из принятых")
        self.btn_export_hl = QPushButton("Экспорт моментов")
        self.btn_export_hl.setObjectName("Primary")
        self.btn_export_timeline = QPushButton("Экспорт склейки")
        self.btn_export_timeline.setObjectName("Primary")
        self.btn_up = QPushButton("↑")
        self.btn_down = QPushButton("↓")
        self.btn_remove = QPushButton("Убрать из таймлайна")
        self.btn_retry = QPushButton("Повторить failed")

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.lbl_help = QLabel(
            "Простой монтаж: текст, лого, музыка, cut. Кандидаты моментов правятся на вкладке «Моменты»."
        )
        self.lbl_help.setObjectName("Subtitle")
        self.lbl_help.setWordWrap(True)

        brand_row = QHBoxLayout()
        brand_row.addWidget(self.btn_logo)
        brand_row.addWidget(self.btn_music)
        brand_row.addWidget(self.btn_color)

        tl_btns = QHBoxLayout()
        tl_btns.addWidget(self.btn_up)
        tl_btns.addWidget(self.btn_down)
        tl_btns.addWidget(self.btn_remove)

        side = QVBoxLayout()
        box = QGroupBox("Брендинг и текст")
        bl = QVBoxLayout(box)
        bl.addLayout(form)
        bl.addLayout(brand_row)
        bl.addWidget(self.btn_save)
        side.addWidget(box)

        side.addWidget(QLabel("Таймлайн"))
        side.addWidget(self.list_timeline, 1)
        side.addLayout(tl_btns)
        side.addWidget(self.btn_build_timeline)
        side.addWidget(self.chk_export_all)
        side.addWidget(self.btn_export_hl)
        side.addWidget(self.btn_export_timeline)
        side.addWidget(QLabel("Очередь рендера"))
        side.addWidget(self.list_jobs, 1)
        side.addWidget(self.btn_retry)
        side.addWidget(self.progress)
        side.addWidget(self.lbl_help)

        root = QHBoxLayout(self)
        root.addWidget(self.player, 3)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setMaximumWidth(400)
        root.addWidget(side_w)

        self.btn_logo.clicked.connect(self._pick_logo)
        self.btn_music.clicked.connect(self._pick_music)
        self.btn_color.clicked.connect(self._pick_color)
        self.btn_save.clicked.connect(self.save_project)
        self.btn_build_timeline.clicked.connect(self.build_timeline)
        self.btn_export_hl.clicked.connect(lambda: self.enqueue_export("highlights"))
        self.btn_export_timeline.clicked.connect(lambda: self.enqueue_export("timeline"))
        self.btn_up.clicked.connect(lambda: self._move_clip(-1))
        self.btn_down.clicked.connect(lambda: self._move_clip(1))
        self.btn_remove.clicked.connect(self._remove_clip)
        self.btn_retry.clicked.connect(self._retry_failed)
        self.list_timeline.currentItemChanged.connect(self._preview_clip)

    def load_project(self, project_id: int) -> None:
        self.project_id = project_id
        self.project = self.db.get_project(project_id)
        if not self.project:
            return
        from app.ui.moments_page import resolve_project_video

        video = resolve_project_video(self.project)
        if video:
            try:
                self.player.open(video)
            except Exception:
                pass
        self.ed_title.setText(self.project.get("title_text") or self.project.get("name") or "")
        self.ed_result.setText(self.project.get("result_text") or "")
        self.ed_logo.setText(self.project.get("logo_path") or "")
        self.ed_music.setText(self.project.get("music_path") or "")
        self.ed_brand.setText(self.project.get("brand_color") or "#C62828")
        self.spin_music.setValue(float(self.project.get("music_volume") or 0.35))
        idx = self.cmb_style.findText(self.project.get("text_style") or "bold")
        if idx >= 0:
            self.cmb_style.setCurrentIndex(idx)
        self.refresh_timeline()
        self.refresh_jobs()

    def save_project(self) -> None:
        if not self.project_id:
            return
        self.db.update_project(
            self.project_id,
            title_text=self.ed_title.text().strip(),
            result_text=self.ed_result.text().strip(),
            logo_path=self.ed_logo.text().strip() or None,
            music_path=self.ed_music.text().strip() or None,
            music_volume=self.spin_music.value(),
            brand_color=self.ed_brand.text().strip() or "#C62828",
            text_style=self.cmb_style.currentText(),
        )
        self.project = self.db.get_project(self.project_id)
        self.status.emit("Настройки монтажа сохранены")

    def build_timeline(self) -> None:
        if not self.project_id:
            return
        accepted = [h for h in self.db.list_highlights(self.project_id) if h.get("accepted")]
        if not accepted:
            QMessageBox.information(self, "Таймлайн", "Нет принятых моментов")
            return
        clips = [
            {
                "highlight_id": h["id"],
                "start_sec": h["start_sec"],
                "end_sec": h["end_sec"],
                "muted": 0,
                "order_index": i,
            }
            for i, h in enumerate(accepted)
        ]
        self.db.replace_timeline(self.project_id, clips)
        self.refresh_timeline()
        self.status.emit(f"Таймлайн: {len(clips)} клипов")

    def refresh_timeline(self) -> None:
        self.list_timeline.clear()
        if not self.project_id:
            return
        for c in self.db.list_timeline(self.project_id):
            item = QListWidgetItem(f"{c['start_sec']:.1f}–{c['end_sec']:.1f}s")
            item.setData(Qt.UserRole, c)
            self.list_timeline.addItem(item)

    def refresh_jobs(self) -> None:
        self.list_jobs.clear()
        if not self.project_id:
            return
        for j in self.db.list_render_jobs(self.project_id):
            text = f"#{j['id']} {j['preset']} · {j['status']} · {int((j.get('progress') or 0)*100)}%"
            if j.get("error"):
                text += f" · {j['error'][:40]}"
            self.list_jobs.addItem(text)

    def _pick_logo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Логотип", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.ed_logo.setText(path)

    def _pick_music(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Музыка", "", "Audio (*.mp3 *.wav *.m4a *.aac)")
        if path:
            self.ed_music.setText(path)

    def _pick_color(self) -> None:
        color = QColorDialog.getColor()
        if color.isValid():
            self.ed_brand.setText(color.name())

    def _timeline_clips(self) -> list[dict]:
        return [self.list_timeline.item(i).data(Qt.UserRole) for i in range(self.list_timeline.count())]

    def _persist_timeline_order(self) -> None:
        clips = []
        for i in range(self.list_timeline.count()):
            data = dict(self.list_timeline.item(i).data(Qt.UserRole))
            data["order_index"] = i
            clips.append(data)
        self.db.replace_timeline(self.project_id, clips)
        self.refresh_timeline()

    def _move_clip(self, delta: int) -> None:
        row = self.list_timeline.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if new_row < 0 or new_row >= self.list_timeline.count():
            return
        item = self.list_timeline.takeItem(row)
        self.list_timeline.insertItem(new_row, item)
        self.list_timeline.setCurrentRow(new_row)
        self._persist_timeline_order()

    def _remove_clip(self) -> None:
        row = self.list_timeline.currentRow()
        if row < 0:
            return
        self.list_timeline.takeItem(row)
        self._persist_timeline_order()

    def _preview_clip(self, current: QListWidgetItem | None, _prev) -> None:
        if not current:
            return
        data = current.data(Qt.UserRole)
        self.player.seek(float(data["start_sec"]))

    def enqueue_export(self, mode: str) -> None:
        if not self.project_id:
            return
        self.save_project()
        preset = self.cmb_preset.currentData()
        payload = {
            "mode": mode,
            "export_all": self.chk_export_all.isChecked(),
        }
        job_id = self.db.create_render_job(self.project_id, preset, payload)
        self.status.emit(f"В очереди задача #{job_id}")
        self.refresh_jobs()
        self._start_worker()

    def _start_worker(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self.progress.setVisible(True)
        self._worker = RenderWorker(self.db)
        self._worker.job_updated.connect(self._on_job_updated)
        self._worker.finished_all.connect(self._on_queue_done)
        self._worker.start()

    def _on_job_updated(self, job_id: int, status: str, progress: float, message: str) -> None:
        self.progress.setValue(int(progress * 100))
        self.lbl_help.setText(f"#{job_id} {status}: {message}")
        self.status.emit(self.lbl_help.text())
        self.refresh_jobs()

    def _on_queue_done(self) -> None:
        self.progress.setVisible(False)
        self.refresh_jobs()
        self.status.emit("Очередь рендера завершена")

    def _retry_failed(self) -> None:
        if not self.project_id:
            return
        for j in self.db.list_render_jobs(self.project_id):
            if j.get("status") == "failed":
                self.db.update_render_job(j["id"], status="queued", progress=0, error=None)
        self.refresh_jobs()
        self._start_worker()
