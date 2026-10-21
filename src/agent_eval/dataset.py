"""Load YAML or JSONL goldens."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from agent_eval.errors import DatasetError
from agent_eval.models import Golden


def load_dataset(path: Path) -> list[Golden]:
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        rows = _load_yaml(path)
    elif suffix == ".jsonl":
        rows = _load_jsonl(path)
    else:
        raise DatasetError(f"dataset {path} must be .yaml, .yml, or .jsonl")
    cases: list[Golden] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise DatasetError(f"{path} row {index} must be a mapping")
        try:
            case = Golden.model_validate(row)
        except ValidationError as exc:
            raise DatasetError(f"{path} row {index} is invalid: {exc}") from exc
        if case.id in seen:
            raise DatasetError(f"duplicate case id {case.id!r} in {path}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError(f"dataset {path} has no cases")
    return cases


def _load_yaml(path: Path) -> list[object]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetError(f"could not parse {path}: {exc}") from exc
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return list(data["cases"])
    raise DatasetError(f"{path} must be a list or a mapping with a cases list")


def _load_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return rows
