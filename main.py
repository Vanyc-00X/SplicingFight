from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.paths import ensure_dirs
from app.styles import APP_STYLESHEET
from app.ui.main_window import MainWindow


def main() -> int:
    ensure_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName("SplicingFight")
    app.setOrganizationName("KarateClub")
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
