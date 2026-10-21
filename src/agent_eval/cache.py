"""File cache for agent runs and metric samples. Keys include a source digest so edits miss."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

from agent_eval.models import AgentResult


class Cache:
    def __init__(self, directory: Path, *, enabled: bool) -> None:
        self.directory = directory
        self.enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    def get_agent(self, key: str) -> AgentResult | None:
        payload = self._read(f"agent-{key}")
        if payload is None:
            return None
        return AgentResult.model_validate(payload)

    def put_agent(self, key: str, result: AgentResult) -> None:
        if result.error:
            return
        self._write(f"agent-{key}", result.model_dump(mode="json"))

    def get_metric(self, key: str) -> dict[str, Any] | None:
        return self._read(f"metric-{key}")

    def put_metric(self, key: str, payload: dict[str, Any]) -> None:
        self._write(f"metric-{key}", payload)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def _read(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _write(self, key: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        temporary = path.with_suffix(".json.tmp")
        encoded = json.dumps(payload, sort_keys=True)
        with self._lock:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)


def digest(*parts: object) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_digest(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def package_digest(module_name: str) -> str:
    """Hash the Python files next to the targeted module so agent edits invalidate the cache."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin or spec.origin == "namespace":
        return "unknown"
    origin = Path(spec.origin)
    root = origin.parent
    hasher = hashlib.sha256()
    files = [origin] if origin.name != "__init__.py" else []
    files.extend(sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts))
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        hasher.update(
            str(resolved.relative_to(root) if root in resolved.parents else resolved.name).encode()
        )
        hasher.update(resolved.read_bytes())
    return hasher.hexdigest()
