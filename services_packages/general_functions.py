# Created by Leandro Carneiro at 13/09/2025
# Description: 
# -------------------------------------------------------------

# services_packages/general_functions.py
from datetime import datetime
from contextlib import contextmanager
import time
import json
from pathlib import Path

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@contextmanager
def stage_timer():
    """Context manager simples para medir duração de uma etapa."""
    t0 = time.perf_counter()
    try:
        yield lambda: (time.perf_counter() - t0)
    finally:
        pass

def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
