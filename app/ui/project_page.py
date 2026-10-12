from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal, QThread, Qt, QDate
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QLineEdit,
    QFormLayout,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QProgressBar,
    QMessageBox,
    QDateEdit,
    QComboBox,
    QSplitter,
    QFrame,
    QDialog,
    QDialogButtonBox,
)

from app.core.ffmpeg_utils import create_proxy, probe_video
from app.db.database import Database
from app.paths import PROJECTS_DIR
from app.ui.widgets.media_pool import MediaPoolWidget, is_video, is_image
from app.ui.widgets.video_player import VideoPlayer


class ImportWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)
    progress = Signal(float, str)

    def __init__(self, source: str, name: str, meta: dict, *, make_proxy: bool = False) -> None:
        super().__init__()
        self.source = source
        self.name = name
        self.meta = meta
        self.make_proxy = make_proxy

    def run(self) -> None:
        try:
            self.progress.emit(0.2, "Чтение метаданных…")
            info = probe_video(self.source)
            proxy_path = None
            if self.make_proxy:
                project_dir = PROJECTS_DIR / f"import_{Path(self.source).stem}"
                project_dir.mkdir(parents=True, exist_ok=True)
                proxy = project_dir / "proxy_720.mp4"
                self.progress.emit(0.35, "Создание proxy 720p…")
                try:
                    create_proxy(self.source, proxy)
                    proxy_path = str(proxy)
                except Exception:
                    proxy_path = None
            self.progress.emit(1.0, "Исходник назначен")
            self.finished_ok.emit(
                {
                    "name": self.name,
                    "source_path": self.source,
                    "proxy_path": proxy_path,
                    **info,
                    **self.meta,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class CreateProjectDialog(QDialog):
    def __init__(self, parent=None, default_folder: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Создать проект")
        self.setMinimumWidth(420)
        self.ed_name = QLineEdit()
        self.ed_name.setPlaceholderText("Например: Кубок весны — финал")
        self.ed_folder = QLineEdit(default_folder or "")
        self.btn_browse = QPushButton("Обзор…")
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.ed_folder, 1)
        folder_row.addWidget(self.btn_browse)
        form = QFormLayout()
        form.addRow("Название проекта", self.ed_name)
        form.addRow("Медиапапка", folder_row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root = QVBoxLayout(self)
        hint = QLabel("Проект как в Resolve: папка = Media Pool, видео назначается из пула.")
        hint.setObjectName("Subtitle")
        hint.setWordWrap(True)
        root.addWidget(hint)
        root.addLayout(form)
        root.addWidget(buttons)
        self.btn_browse.clicked.connect(self._browse)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Медиапапка проекта", self.ed_folder.text())
        if path:
            self.ed_folder.setText(path)

    def values(self) -> tuple[str, str]:
        return self.ed_name.text().strip(), self.ed_folder.text().strip()


class ProjectPage(QWidget):
    """DaVinci Resolve-inspired layout: Media Pool | Viewer | Inspector + project bin."""

    project_opened = Signal(int)
    status = Signal(str)

    def __init__(self, db: Database, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self._worker: ImportWorker | None = None
        self.current_project_id: int | None = None
        self._preview_path: str | None = None

        # --- Toolbar ---
        self.btn_create = QPushButton("+ Создать проект")
        self.btn_create.setObjectName("Primary")
        self.btn_open_proj = QPushButton("Открыть")
        self.btn_delete = QPushButton("Удалить")
        self.btn_use_media = QPushButton("Назначить исходником")
        self.btn_use_media.setObjectName("Success")
        self.btn_import_file = QPushButton("Импорт файла…")
        self.lbl_project = QLabel("Нет активного проекта")
        self.lbl_project.setObjectName("PanelTitle")

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 6, 8, 6)
        toolbar.addWidget(self.btn_create)
        toolbar.addWidget(self.btn_open_proj)
        toolbar.addWidget(self.btn_delete)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.btn_use_media)
        toolbar.addWidget(self.btn_import_file)
        toolbar.addStretch(1)
        toolbar.addWidget(self.lbl_project)
        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("Toolbar")
        toolbar_frame.setLayout(toolbar)

        # --- Media Pool ---
        self.media_pool = MediaPoolWidget()

        # --- Viewer ---
        self.player = VideoPlayer()
        viewer_wrap = QFrame()
        viewer_wrap.setObjectName("Panel")
        vl = QVBoxLayout(viewer_wrap)
        vl.setContentsMargins(8, 8, 8, 8)
        viewer_title = QLabel("Viewer")
        viewer_title.setObjectName("PanelTitle")
        self.lbl_viewer_info = QLabel("Двойной клик по клипу в Media Pool — превью")
        self.lbl_viewer_info.setObjectName("Subtitle")
        vl.addWidget(viewer_title)
        vl.addWidget(self.player, 1)
        vl.addWidget(self.lbl_viewer_info)

        # --- Inspector ---
        self.ed_name = QLineEdit()
        self.ed_event = QLineEdit()
        self.ed_red = QLineEdit()
        self.ed_blue = QLineEdit()
        self.ed_category = QLineEdit()
        self.date_event = QDateEdit(QDate.currentDate())
        self.date_event.setCalendarPopup(True)
        self.cmb_target = QComboBox()
        self.cmb_target.addItems(["red", "blue"])
        self.btn_save = QPushButton("Сохранить")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.lbl_info = QLabel("Создайте проект и откройте папку с медиа")
        self.lbl_info.setObjectName("Subtitle")
        self.lbl_info.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Название", self.ed_name)
        form.addRow("Турнир", self.ed_event)
        form.addRow("Дата", self.date_event)
        form.addRow("Красный", self.ed_red)
        form.addRow("Синий", self.ed_blue)
        form.addRow("Категория", self.ed_category)
        form.addRow("Трекинг", self.cmb_target)

        inspector = QFrame()
        inspector.setObjectName("Panel")
        il = QVBoxLayout(inspector)
        il.setContentsMargins(8, 8, 8, 8)
        insp_title = QLabel("Inspector")
        insp_title.setObjectName("PanelTitle")
        il.addWidget(insp_title)
        il.addLayout(form)
        il.addWidget(self.btn_save)
        il.addWidget(self.progress)
        il.addWidget(self.lbl_info)
        il.addStretch(1)

        # --- Project bin (bottom) ---
        self.list_projects = QListWidget()
        self.list_projects.setObjectName("ProjectBin")
        self.list_projects.setMaximumHeight(130)
        self.list_projects.itemDoubleClicked.connect(self._open_selected)
        bin_title = QLabel("Project Bin — проекты клуба")
        bin_title.setObjectName("PanelTitle")
        bin_frame = QFrame()
        bin_frame.setObjectName("Panel")
        bl = QVBoxLayout(bin_frame)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.addWidget(bin_title)
        bl.addWidget(self.list_projects)

        # --- Splitters (Resolve-like) ---
        top_split = QSplitter(Qt.Horizontal)
        top_split.setObjectName("MainSplit")
        pool_frame = QFrame()
        pool_frame.setObjectName("Panel")
        pl = QVBoxLayout(pool_frame)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.media_pool)
        top_split.addWidget(pool_frame)
        top_split.addWidget(viewer_wrap)
        top_split.addWidget(inspector)
        top_split.setStretchFactor(0, 3)
        top_split.setStretchFactor(1, 5)
        top_split.setStretchFactor(2, 2)
        top_split.setSizes([320, 640, 280])

        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(top_split)
        main_split.addWidget(bin_frame)
        main_split.setStretchFactor(0, 5)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([600, 140])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(toolbar_frame)
        root.addWidget(main_split, 1)

        # signals
        self.btn_create.clicked.connect(self.create_project)
        self.btn_open_proj.clicked.connect(self._open_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_use_media.clicked.connect(self.assign_selected_media)
        self.btn_import_file.clicked.connect(self.import_video)
        self.btn_save.clicked.connect(self.save_meta)
        self.media_pool.media_activated.connect(self.preview_media)
        self.media_pool.media_selected.connect(self._on_media_selected)
        self.media_pool.folder_changed.connect(self._on_folder_changed)
        # Single-click also loads Viewer (Resolve-like)

        self.refresh()

    # ----- project CRUD -----

    def create_project(self) -> None:
        dlg = CreateProjectDialog(self, self.media_pool.folder)
        if dlg.exec() != QDialog.Accepted:
            return
        name, folder = dlg.values()
        if not name:
            QMessageBox.warning(self, "Проект", "Укажите название")
            return
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Проект", "Выберите существующую медиапапку")
            return
        pid = self.db.create_project(
            name,
            source_path="",
            media_folder=folder,
            event_date=self.date_event.date().toString("yyyy-MM-dd"),
        )
        self.refresh()
        self.load_project(pid)
        self.status.emit(f"Проект создан: {name}")

    def refresh(self) -> None:
        self.list_projects.clear()
        for p in self.db.list_projects():
            folder = p.get("media_folder") or ""
            src = Path(p.get("source_path") or "").name
            label = f"{p['name']}"
            if src:
                label += f"  ·  {src}"
            elif folder:
                label += f"  ·  {Path(folder).name}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, p["id"])
            item.setToolTip(folder or p.get("source_path") or "")
            self.list_projects.addItem(item)

    def _open_selected(self) -> None:
        item = self.list_projects.currentItem()
        if not item:
            QMessageBox.information(self, "Проект", "Выберите проект в Project Bin")
            return
        self.load_project(int(item.data(Qt.UserRole)))

    def load_project(self, project_id: int) -> None:
        p = self.db.get_project(project_id)
        if not p:
            return
        self.current_project_id = project_id
        self.lbl_project.setText(f"● {p.get('name')}")
        self.ed_name.setText(p.get("name") or "")
        self.ed_event.setText(p.get("event_name") or "")
        self.ed_red.setText(p.get("fighter_red") or "")
        self.ed_blue.setText(p.get("fighter_blue") or "")
        self.ed_category.setText(p.get("category") or "")
        if p.get("event_date"):
            self.date_event.setDate(QDate.fromString(p["event_date"], "yyyy-MM-dd"))
        idx = self.cmb_target.findText(p.get("target_athlete") or "red")
        if idx >= 0:
            self.cmb_target.setCurrentIndex(idx)

        folder = p.get("media_folder")
        if folder and Path(folder).is_dir():
            self.media_pool.set_folder(folder)
        elif p.get("source_path"):
            parent = str(Path(p["source_path"]).parent)
            if Path(parent).is_dir():
                self.media_pool.set_folder(parent)

        video = p.get("proxy_path") or p.get("source_path")
        if video and Path(video).exists():
            try:
                self.player.open(video)
                self._preview_path = video
                self.lbl_viewer_info.setText(Path(video).name)
            except Exception as exc:
                QMessageBox.warning(self, "Видео", str(exc))
        else:
            self.lbl_viewer_info.setText("Исходник не назначен — выберите видео в Media Pool")

        self.lbl_info.setText(
            f"{int(p.get('width') or 0)}×{int(p.get('height') or 0)} · "
            f"{p.get('fps', 0):.1f} fps · {p.get('duration', 0):.1f} c"
            if p.get("source_path")
            else "Назначьте видео исходником (кнопка выше или двойной клик + назначить)"
        )
        self.project_opened.emit(project_id)

    def delete_selected(self) -> None:
        item = self.list_projects.currentItem()
        if not item:
            return
        pid = int(item.data(Qt.UserRole))
        if (
            QMessageBox.question(
                self,
                "Удалить?",
                "Удалить проект из базы?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.db.delete_project(pid)
        if self.current_project_id == pid:
            self.current_project_id = None
            self.lbl_project.setText("Нет активного проекта")
        self.refresh()

    def save_meta(self) -> None:
        if not self.current_project_id:
            QMessageBox.information(self, "Проект", "Сначала создайте или откройте проект")
            return
        self.db.update_project(
            self.current_project_id,
            name=self.ed_name.text().strip() or "Без названия",
            event_name=self.ed_event.text().strip() or None,
            event_date=self.date_event.date().toString("yyyy-MM-dd"),
            fighter_red=self.ed_red.text().strip() or None,
            fighter_blue=self.ed_blue.text().strip() or None,
            category=self.ed_category.text().strip() or None,
            target_athlete=self.cmb_target.currentText(),
            media_folder=self.media_pool.folder,
        )
        self.refresh()
        self.lbl_project.setText(f"● {self.ed_name.text().strip()}")
        self.status.emit("Метаданные сохранены")

    # ----- media pool -----

    def _on_folder_changed(self, folder: str) -> None:
        if self.current_project_id:
            self.db.update_project(self.current_project_id, media_folder=folder)
        self.status.emit(f"Media Pool: {folder}")

    def _on_media_selected(self, path: str) -> None:
        kind = "видео" if is_video(Path(path)) else "фото"
        self.lbl_info.setText(f"Выбрано ({kind}): {Path(path).name}")
        # Auto-preview on selection so Viewer is never empty after a click
        self.preview_media(path)

    def preview_media(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(self, "Превью", f"Файл не найден:\n{path}")
            return
        if is_video(p):
            try:
                self.player.open(str(p))
                self._preview_path = str(p)
                dur = self.player.duration
                self.lbl_viewer_info.setText(
                    f"{p.name}  ·  {self.player.frame_size[0]}x{self.player.frame_size[1]}  ·  {dur:.1f}s"
                )
                self.status.emit(f"Превью: {p.name}")
                # Auto-bind source to active project so Cut never stays empty
                if self.current_project_id:
                    proj = self.db.get_project(self.current_project_id)
                    if proj and (not proj.get("source_path") or not Path(proj["source_path"]).exists()):
                        self._quick_bind_source(str(p))
            except Exception as exc:
                QMessageBox.warning(self, "Превью", str(exc))
        elif is_image(p):
            self._show_image_preview(str(p))
        else:
            QMessageBox.information(self, "Медиа", "Неподдерживаемый формат")

    def _quick_bind_source(self, path: str) -> None:
        """Bind source immediately (no thread) so Cut tab can open the same file."""
        try:
            info = probe_video(path)
            self.db.update_project(
                self.current_project_id,
                source_path=path,
                duration=info.get("duration", 0),
                fps=info.get("fps", 25),
                width=int(info.get("width") or 0),
                height=int(info.get("height") or 0),
                media_folder=str(Path(path).parent),
                name=self.ed_name.text().strip() or Path(path).stem,
                event_name=self.ed_event.text().strip() or None,
                fighter_red=self.ed_red.text().strip() or None,
                fighter_blue=self.ed_blue.text().strip() or None,
                category=self.ed_category.text().strip() or None,
                target_athlete=self.cmb_target.currentText(),
            )
            self.refresh()
            self.lbl_info.setText(
                f"Исходник: {Path(path).name} · "
                f"{int(info.get('width') or 0)}x{int(info.get('height') or 0)} · "
                f"{info.get('duration', 0):.1f}s"
            )
            self.project_opened.emit(self.current_project_id)
            self.status.emit(f"Исходник автоматически привязан: {Path(path).name}")
        except Exception as exc:
            self.status.emit(f"Не удалось привязать исходник: {exc}")

    def _show_image_preview(self, path: str) -> None:
        import cv2
        import numpy as np

        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Фото", "Не удалось открыть изображение")
            return
        self._preview_path = path
        self.lbl_viewer_info.setText(Path(path).name)
        # show still via player canvas
        self.player.canvas.show_frame(img)
        self.player.lbl_time.setText("фото")
        self.status.emit(f"Превью фото: {Path(path).name}")

    def assign_selected_media(self) -> None:
        path = self.media_pool.selected_path() or self._preview_path
        if not path:
            QMessageBox.information(self, "Исходник", "Выберите видео в Media Pool")
            return
        if not is_video(Path(path)):
            QMessageBox.information(self, "Исходник", "Исходником боя может быть только видео")
            return
        if not self.current_project_id:
            name = self.ed_name.text().strip() or Path(path).stem
            pid = self.db.create_project(
                name,
                source_path="",
                media_folder=str(Path(path).parent),
                event_date=self.date_event.date().toString("yyyy-MM-dd"),
                event_name=self.ed_event.text().strip() or None,
                fighter_red=self.ed_red.text().strip() or None,
                fighter_blue=self.ed_blue.text().strip() or None,
                category=self.ed_category.text().strip() or None,
            )
            self.current_project_id = pid
            self.refresh()
        # Fast assign: no proxy wait — Cut works immediately
        self._start_import(path, make_proxy=False)

    def import_video(self) -> None:
        start = self.media_pool.folder or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Импорт видео",
            start,
            "Video (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;All (*.*)",
        )
        if not path:
            return
        if not self.current_project_id:
            name = self.ed_name.text().strip() or Path(path).stem
            pid = self.db.create_project(
                name,
                source_path="",
                media_folder=str(Path(path).parent),
                event_date=self.date_event.date().toString("yyyy-MM-dd"),
            )
            self.current_project_id = pid
            self.media_pool.set_folder(str(Path(path).parent))
            self.refresh()
        self._start_import(path, make_proxy=False)

    def _start_import(self, path: str, *, make_proxy: bool = False) -> None:
        name = self.ed_name.text().strip() or Path(path).stem
        self.ed_name.setText(name)
        meta = {
            "event_date": self.date_event.date().toString("yyyy-MM-dd"),
            "event_name": self.ed_event.text().strip() or None,
            "fighter_red": self.ed_red.text().strip() or None,
            "fighter_blue": self.ed_blue.text().strip() or None,
            "category": self.ed_category.text().strip() or None,
            "media_folder": self.media_pool.folder or str(Path(path).parent),
        }
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.btn_use_media.setEnabled(False)
        self.btn_import_file.setEnabled(False)
        self._worker = ImportWorker(path, name, meta, make_proxy=make_proxy)
        self._worker.progress.connect(self._on_import_progress)
        self._worker.finished_ok.connect(self._on_import_done)
        self._worker.failed.connect(self._on_import_fail)
        self._worker.start()

    def _on_import_progress(self, value: float, msg: str) -> None:
        self.progress.setValue(int(value * 100))
        self.lbl_info.setText(msg)
        self.status.emit(msg)

    def _on_import_done(self, data: dict) -> None:
        self.btn_use_media.setEnabled(True)
        self.btn_import_file.setEnabled(True)
        self.progress.setVisible(False)
        if not self.current_project_id:
            return
        self.db.update_project(
            self.current_project_id,
            name=data["name"],
            source_path=data["source_path"],
            proxy_path=data.get("proxy_path"),
            duration=data.get("duration", 0),
            fps=data.get("fps", 25),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            event_date=data.get("event_date"),
            event_name=data.get("event_name"),
            fighter_red=data.get("fighter_red"),
            fighter_blue=data.get("fighter_blue"),
            category=data.get("category"),
            media_folder=data.get("media_folder") or self.media_pool.folder,
            target_athlete=self.cmb_target.currentText(),
        )
        self.refresh()
        self.load_project(self.current_project_id)
        self.status.emit(f"Исходник назначен: {Path(data['source_path']).name}")

    def _on_import_fail(self, err: str) -> None:
        self.btn_use_media.setEnabled(True)
        self.btn_import_file.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Ошибка импорта", err)
