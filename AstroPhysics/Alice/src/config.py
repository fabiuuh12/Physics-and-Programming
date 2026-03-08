from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AliceConfig:
    project_root: Path
    allowed_roots: list[Path]
    log_dir: Path
    max_runtime_seconds: int


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return (base_dir / path).resolve()
    return path.resolve()


def load_config(config_path: Path) -> AliceConfig:
    config_path = config_path.expanduser().resolve()
    project_root = config_path.parent.parent

    data: dict[str, object] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    allowed_raw = data.get("allowed_roots", [str(project_root)])
    if not isinstance(allowed_raw, list) or not allowed_raw:
        allowed_raw = [str(project_root)]
    allowed_roots = [_resolve_path(str(item), project_root) for item in allowed_raw]

    log_raw = data.get("log_dir", "logs")
    log_dir = _resolve_path(str(log_raw), project_root)

    max_runtime = data.get("max_runtime_seconds", 300)
    if not isinstance(max_runtime, int) or max_runtime <= 0:
        max_runtime = 300

    return AliceConfig(
        project_root=project_root,
        allowed_roots=allowed_roots,
        log_dir=log_dir,
        max_runtime_seconds=max_runtime,
    )
