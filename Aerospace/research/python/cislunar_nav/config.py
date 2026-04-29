"""Config helpers for cislunar navigation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def output_directory(config: dict[str, Any], run_name: str | None = None) -> Path:
    root = Path(config["output"]["directory"])
    out = root / run_name if run_name else root
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_config_snapshot(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
