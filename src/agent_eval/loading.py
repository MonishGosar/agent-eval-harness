"""Import `module:attribute` targets and put the project root on sys.path."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_eval.config import find_project_root
from agent_eval.errors import AdapterError, ConfigError


def prepare_import_path(config_path: Path) -> Path:
    root = find_project_root(config_path if config_path.is_dir() else config_path.parent)
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root


def load_target(spec: str) -> Any:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ConfigError(f"target must look like 'package.module:attribute', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AdapterError(f"could not import {module_name!r} from target {spec!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise AdapterError(f"{module_name!r} has no attribute {attribute!r}") from exc


def load_callable(spec: str) -> Callable[..., Any]:
    target = load_target(spec)
    if not callable(target):
        raise AdapterError(f"target {spec!r} is not callable")
    return target
