"""Configuration helpers for ASF experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment JSON config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def output_directory(config: dict[str, Any]) -> Path:
    """Return and create the configured output directory."""
    out = Path(config["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    return out
