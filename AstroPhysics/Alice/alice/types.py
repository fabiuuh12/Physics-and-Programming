from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Intent:
    action: str
    target: Optional[str] = None
    pid: Optional[int] = None
    requires_confirmation: bool = False
    raw: str = ""


@dataclass
class ExecResult:
    ok: bool = False
    message: str = ""


@dataclass
class MemoryItem:
    id: int = 0
    content: str = ""
    category: str = ""
    created_at: str = ""
    use_count: int = 0
