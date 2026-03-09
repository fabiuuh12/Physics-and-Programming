from __future__ import annotations

from pathlib import Path

from .string_utils import normalize_text, now_iso8601, split_words, trim
from .types import MemoryItem


class MemoryStore:
    def __init__(self, db_path: Path):
        self._db_path = db_path.expanduser().resolve(strict=False)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._items: list[MemoryItem] = []
        self._next_id = 1
        self._load()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def count(self) -> int:
        return len(self._items)

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")

    @staticmethod
    def _unescape(value: str) -> str:
        out: list[str] = []
        i = 0
        while i < len(value):
            if value[i] != "\\" or i + 1 >= len(value):
                out.append(value[i])
                i += 1
                continue
            nxt = value[i + 1]
            if nxt == "t":
                out.append("\t")
            elif nxt == "n":
                out.append("\n")
            else:
                out.append(nxt)
            i += 2
        return "".join(out)

    def _load(self) -> None:
        self._items.clear()
        self._next_id = 1

        try:
            lines = self._db_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return

        for line in lines:
            if not trim(line):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                item_id = int(parts[0])
                use_count = int(parts[1])
            except ValueError:
                continue

            item = MemoryItem(
                id=item_id,
                use_count=use_count,
                created_at=self._unescape(parts[2]),
                category=self._unescape(parts[3]),
                content=self._unescape(parts[4]),
            )
            self._items.append(item)
            self._next_id = max(self._next_id, item.id + 1)

    def _save(self) -> bool:
        lines: list[str] = []
        for item in self._items:
            lines.append(
                f"{item.id}\t{item.use_count}\t{self._escape(item.created_at)}\t"
                f"{self._escape(item.category)}\t{self._escape(item.content)}"
            )
        try:
            self._db_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            return True
        except OSError:
            return False

    def add(self, content: str, category: str = "general") -> bool:
        cleaned = trim(content)
        if not cleaned:
            return False
        normalized_new = normalize_text(cleaned)
        if not normalized_new:
            return False

        for item in self._items:
            if normalize_text(item.content) == normalized_new:
                return False

        item = MemoryItem(
            id=self._next_id,
            content=cleaned,
            category=category,
            created_at=now_iso8601(),
            use_count=0,
        )
        self._next_id += 1
        self._items.append(item)
        return self._save()

    def recent(self, limit: int = 5) -> list[MemoryItem]:
        if limit <= 0:
            limit = 5
        start = max(0, len(self._items) - limit)
        return list(reversed(self._items[start:]))

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        if limit <= 0:
            limit = 5

        q = normalize_text(query)
        if not q:
            return self.recent(limit)

        q_tokens = split_words(q)
        scored: list[tuple[int, int]] = []
        for idx, item in enumerate(self._items):
            normalized = normalize_text(item.content)
            score = 0
            if q in normalized:
                score += 8
            for token in q_tokens:
                if len(token) < 3:
                    continue
                if token in normalized:
                    score += 2
            score += min(item.use_count, 5)
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        result: list[MemoryItem] = []
        for _, idx in scored[:limit]:
            self._items[idx].use_count += 1
            result.append(self._items[idx])

        if result:
            self._save()
        return result
