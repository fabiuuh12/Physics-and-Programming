from __future__ import annotations

import datetime as _dt
import re
import shlex
from pathlib import Path
from typing import Optional


def trim(value: str) -> str:
    return value.strip()


def to_lower(value: str) -> str:
    return value.lower()


def normalize_text(value: str) -> str:
    lowered = to_lower(trim(value))
    out: list[str] = []
    previous_space = False
    for ch in lowered:
        if ch.isalnum() or ch in {"_", "'"}:
            out.append(ch)
            previous_space = False
            continue
        if not previous_space:
            out.append(" ")
            previous_space = True
    return "".join(out).strip()


def split_words(value: str) -> list[str]:
    return value.split()


def starts_with_ci(text: str, prefix: str) -> bool:
    return text.lower().startswith(prefix.lower())


def replace_all(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def join(parts: list[str], delim: str) -> str:
    return delim.join(parts)


def now_iso8601() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def format_clock_time() -> str:
    return _dt.datetime.now().strftime("%I:%M %p")


def format_long_date() -> str:
    return _dt.datetime.now().strftime("%A, %B %d, %Y")


def read_file(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def write_file(path: str, content: str) -> bool:
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return True
    except OSError:
        return False


def collapse_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
