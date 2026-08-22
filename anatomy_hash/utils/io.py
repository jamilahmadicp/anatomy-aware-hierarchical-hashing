from __future__ import annotations
from pathlib import Path
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
import torch


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def environment_snapshot():
    snap = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        snap["gpu"] = torch.cuda.get_device_name(0)
    try:
        snap["pip_freeze"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    except Exception:
        snap["pip_freeze"] = []
    return snap
