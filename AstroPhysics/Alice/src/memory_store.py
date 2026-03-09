from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class MemoryItem:
    id: int
    content: str
    category: str
    created_at: str
    use_count: int


def _normalize_text(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _tokenize(value: str) -> list[str]:
    return [token for token in _normalize_text(value).split() if len(token) >= 3]


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path.expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    normalized TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL DEFAULT 'general',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    use_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.commit()

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
            return int(row["n"]) if row is not None else 0

    def add(self, content: str, *, category: str = "general") -> bool:
        cleaned = " ".join(content.strip().split())
        if not cleaned:
            return False
        normalized = _normalize_text(cleaned)
        if not normalized:
            return False

        timestamp = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO memories (content, normalized, category, created_at, last_used_at, use_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (cleaned, normalized, category, timestamp, timestamp, 0),
                )
                connection.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def _touch(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE memories
                SET use_count = use_count + 1,
                    last_used_at = ?
                WHERE id IN ({placeholders})
                """,
                [timestamp, *memory_ids],
            )
            connection.commit()

    def recent(self, *, limit: int = 5) -> list[MemoryItem]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, content, category, created_at, use_count
                FROM memories
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            MemoryItem(
                id=int(row["id"]),
                content=str(row["content"]),
                category=str(row["category"]),
                created_at=str(row["created_at"]),
                use_count=int(row["use_count"]),
            )
            for row in rows
        ]

    def search(self, query: str, *, limit: int = 5) -> list[MemoryItem]:
        trimmed = query.strip()
        if not trimmed:
            return self.recent(limit=limit)

        tokens = _tokenize(trimmed)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, content, category, created_at, use_count
                FROM memories
                ORDER BY datetime(created_at) DESC
                LIMIT 300
                """
            ).fetchall()

        scored: list[tuple[int, MemoryItem]] = []
        query_norm = _normalize_text(trimmed)
        for row in rows:
            content = str(row["content"])
            normalized = _normalize_text(content)
            score = 0
            if query_norm and query_norm in normalized:
                score += 8
            for token in tokens:
                if token in normalized:
                    score += 2
            if score == 0:
                continue
            scored.append(
                (
                    score + min(int(row["use_count"]), 5),
                    MemoryItem(
                        id=int(row["id"]),
                        content=content,
                        category=str(row["category"]),
                        created_at=str(row["created_at"]),
                        use_count=int(row["use_count"]),
                    ),
                )
            )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = [item for _, item in scored[:limit]]
        self._touch([item.id for item in selected])
        return selected
