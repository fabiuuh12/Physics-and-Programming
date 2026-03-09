from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AliceConfig:
    project_root: Path
    allowed_roots: list[Path]
    log_dir: Path
    max_runtime_seconds: int = 300


def _resolve_path(raw: str, base: Path) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.expanduser().resolve(strict=False)


def load_config(config_path_raw: Path) -> AliceConfig:
    config_path = config_path_raw.expanduser().resolve(strict=False)
    project_root = config_path.parent.parent.resolve(strict=False)

    text = ""
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        pass

    data: dict = {}
    if text:
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            data = {}

    allowed_roots_raw = data.get("allowed_roots", [])
    allowed_roots: list[Path] = []
    if isinstance(allowed_roots_raw, list):
        for item in allowed_roots_raw:
            if isinstance(item, str) and item.strip():
                allowed_roots.append(_resolve_path(item, project_root))
    if not allowed_roots:
        allowed_roots = [project_root]

    log_dir = _resolve_path(str(data.get("log_dir", "logs")), project_root)

    max_runtime_seconds = 300
    raw_runtime = data.get("max_runtime_seconds")
    if isinstance(raw_runtime, int) and raw_runtime > 0:
        max_runtime_seconds = raw_runtime

    return AliceConfig(
        project_root=project_root,
        allowed_roots=allowed_roots,
        log_dir=log_dir,
        max_runtime_seconds=max_runtime_seconds,
    )
