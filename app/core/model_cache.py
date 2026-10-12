from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.device import torch_device, ultralytics_kwargs
from app.paths import MODELS_DIR, ensure_dirs

_YOLO: dict[str, Any] = {}


def resolve_weights(name: str) -> str:
    ensure_dirs()
    local = MODELS_DIR / name
    if local.exists():
        return str(local)
    # Also accept cwd-downloaded weights
    cwd = Path(name)
    if cwd.exists():
        try:
            cwd.replace(local)
            return str(local)
        except Exception:
            return str(cwd)
    return str(local)


def get_yolo(model_name: str = "yolov8n.pt"):
    """Singleton YOLO model — avoid reloading weights every track/highlight run."""
    key = model_name
    if key in _YOLO:
        return _YOLO[key]
    from ultralytics import YOLO

    path = resolve_weights(model_name)
    try:
        model = YOLO(path)
    except Exception:
        model = YOLO(resolve_weights("yolov8n.pt"))
        key = "yolov8n.pt"
    # Keep weights on GPU; warm-up once so first real call is fast
    try:
        import numpy as np
        import torch

        if torch_device() == "cuda":
            try:
                model.to("cuda")
            except Exception:
                pass
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        model.predict(dummy, **ultralytics_kwargs(imgsz=512))
        if torch_device() == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass
    _YOLO[key] = model
    return model


def preload_tracker_model() -> str:
    """Eager-load detect weights so first track click is fast."""
    get_yolo("yolov8n.pt")
    return yolo_device_info()


def reset_tracker_state(model) -> None:
    """Reset ByteTrack state without unloading CUDA weights."""
    try:
        if getattr(model, "predictor", None) is not None:
            model.predictor = None
    except Exception:
        pass


def yolo_device_info() -> str:
    from app.core.device import device_label

    return device_label()
