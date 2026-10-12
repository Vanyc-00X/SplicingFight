from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QStatusBar,
    QFrame,
)

from app.db.database import Database
from app.ui.editor_page import EditorPage
from app.ui.library_page import LibraryPage
from app.ui.moments_page import MomentsPage
from app.ui.project_page import ProjectPage


class _ModelWarmupWorker(QThread):
    done = Signal(str)

    def run(self) -> None:
        try:
            from app.core.model_cache import preload_tracker_model

            label = preload_tracker_model()
            self.done.emit(label)
        except Exception:
            self.done.emit("CPU")


class MainWindow(QMainWindow):
    def __init__(self, db: Database | None = None) -> None:
        super().__init__()
        self.db = db or Database()
        self.setWindowTitle("SplicingFight — бои каратэ для SMM")
        self.resize(1440, 900)
        self._current_project_id: int | None = None
        self._warmup: _ModelWarmupWorker | None = None

        header = QFrame()
        header.setObjectName("Toolbar")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 8, 12, 8)
        brand = QLabel("SplicingFight")
        brand.setObjectName("BrandTitle")
        subtitle = QLabel(
            "Media Pool · Viewer · Inspector · Timeline  ·  локальный NLE для каратэ"
        )
        subtitle.setObjectName("Subtitle")
        hl.addWidget(brand)
        hl.addSpacing(16)
        hl.addWidget(subtitle, 1)

        self.tabs = QTabWidget()
        self.page_project = ProjectPage(self.db)
        self.page_moments = MomentsPage(self.db)
        self.page_editor = EditorPage(self.db)
        self.page_library = LibraryPage(self.db)

        self.tabs.addTab(self.page_project, "Media")
        self.tabs.addTab(self.page_moments, "Cut / Моменты")
        self.tabs.addTab(self.page_editor, "Edit / Экспорт")
        self.tabs.addTab(self.page_library, "Deliver / Библиотека")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Готово · Media: назначьте исходник → Cut: трекинг и моменты"
        )

        self.page_project.project_opened.connect(self._on_project_opened)
        self.page_project.status.connect(self._set_status)
        self.page_moments.status.connect(self._set_status)
        self.page_editor.status.connect(self._set_status)
        self.page_library.status.connect(self._set_status)
        self.page_moments.highlights_changed.connect(self.page_editor.refresh_timeline)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        QTimer.singleShot(80, self._start_model_warmup)

    def _start_model_warmup(self) -> None:
        self._set_status("Загрузка YOLO на GPU…")
        self._warmup = _ModelWarmupWorker(self)
        self._warmup.done.connect(self._on_warmup_done)
        self._warmup.start()

    def _on_warmup_done(self, label: str) -> None:
        self._set_status(f"Трекер готов · {label}")

    def _on_project_opened(self, project_id: int) -> None:
        self._current_project_id = project_id
        self.page_moments.load_project(project_id)
        self.page_editor.load_project(project_id)
        project = self.db.get_project(project_id)
        name = project.get("name") if project else ""
        src = project.get("source_path") if project else ""
        if src and Path(src).exists():
            self._set_status(f"Проект «{name}» · исходник готов · можно Cut / Моменты")
        else:
            self._set_status(
                f"Проект «{name}» · выберите видео и нажмите «Назначить исходником»"
            )

    def _on_tab_changed(self, index: int) -> None:
        # Keep Cut/Edit in sync when user switches tabs
        pid = self._current_project_id or self.page_project.current_project_id
        if not pid:
            return
        self._current_project_id = pid
        widget = self.tabs.widget(index)
        if widget is self.page_moments:
            self.page_moments.load_project(pid)
        elif widget is self.page_editor:
            self.page_editor.load_project(pid)
        elif widget is self.page_library:
            self.page_library.refresh()

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)
