from __future__ import annotations

from pathlib import Path

APP_NAME = "SplicingFight"
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path.home() / "AppData" / "Local" / APP_NAME
PROJECTS_DIR = DATA_DIR / "projects"
LIBRARY_DIR = DATA_DIR / "library"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "splicingfight.db"


def ensure_dirs() -> None:
    for path in (DATA_DIR, PROJECTS_DIR, LIBRARY_DIR, CACHE_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
