from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .config import JOBS_DIR

_LOCK = threading.RLock()
_JOBS: dict[str, dict[str, Any]] = {}


def _metadata_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / "job.json"


def put(job_id: str, value: dict[str, Any]) -> None:
    with _LOCK:
        _JOBS[job_id] = value
        folder = JOBS_DIR / job_id
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder / "job.json.tmp"
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(_metadata_path(job_id))


def get(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        if job_id in _JOBS:
            return dict(_JOBS[job_id])
        path = _metadata_path(job_id)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        _JOBS[job_id] = value
        return dict(value)


def update(job_id: str, **changes: Any) -> dict[str, Any]:
    with _LOCK:
        current = get(job_id)
        if current is None:
            raise KeyError(job_id)
        current.update(changes)
        put(job_id, current)
        return dict(current)


def public(value: dict[str, Any]) -> dict[str, Any]:
    hidden = {"video_path", "audio_path", "reference_path", "reference_text", "job_dir"}
    return {key: item for key, item in value.items() if key not in hidden}
