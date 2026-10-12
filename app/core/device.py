from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def torch_device() -> str:
    """Return 'cuda' if usable GPU is available, else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            # touch device once
            _ = torch.zeros(1, device="cuda")
            return "cuda"
    except Exception:
        pass
    return "cpu"


def ultralytics_kwargs(imgsz: int = 640) -> dict:
    """Common predict/track kwargs — pin YOLO to CUDA device 0 when available."""
    device = torch_device()
    return {
        "device": 0 if device == "cuda" else "cpu",
        "imgsz": imgsz,
        "verbose": False,
    }


def device_label() -> str:
    dev = torch_device()
    if dev != "cuda":
        return "CPU"
    try:
        import torch

        return f"CUDA:{torch.cuda.get_device_name(0)}"
    except Exception:
        return "CUDA"
