from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3

from .string_utils import normalize_text, now_iso8601, split_words, trim
from .types import AutonomyLearningItem, MemoryItem


class MemoryStore:
    _SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}

    def __init__(self, db_path: Path):
        requested_path = db_path.expanduser().resolve(strict=False)
        self._legacy_tsv_path: Path | None = None

        suffix = requested_path.suffix.lower()
        if suffix == ".tsv":
            self._legacy_tsv_path = requested_path
            self._db_path = requested_path.with_suffix(".db")
        elif suffix in self._SQLITE_SUFFIXES:
            self._db_path = requested_path
            candidate_tsv = requested_path.with_suffix(".tsv")
            if candidate_tsv.exists():
                self._legacy_tsv_path = candidate_tsv
        else:
            self._db_path = requested_path.with_suffix(".db")

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, timeout=10)
        self._conn.row_factory = sqlite3.Row

        # Better write/read behavior for frequent small updates.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._ensure_schema()
        self._backfill_metadata()
        self._maybe_import_legacy_tsv()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.close()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()
        return int(row["c"]) if row is not None else 0

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

    @staticmethod
    def _normalize_key(key: str) -> str:
        raw = normalize_text(key)
        raw = re.sub(r"[^a-z0-9._-]+", "_", raw)
        raw = re.sub(r"_+", "_", raw).strip("_")
        return raw[:64]

    @staticmethod
    def _structured_content(key: str, value: str) -> str:
        return f"[{key}] {value}"

    @staticmethod
    def _parse_structured(content: str) -> tuple[str, str] | None:
        m = re.match(r"^\[([a-z0-9._-]{1,64})\]\s+(.+)$", trim(content), flags=re.IGNORECASE)
        if not m:
            return None
        return m.group(1).lower(), trim(m.group(2))

    @classmethod
    def _key_from_content(cls, content: str) -> str | None:
        parsed = cls._parse_structured(content)
        if not parsed:
            return None
        return parsed[0]

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        aa = set(split_words(normalize_text(a)))
        bb = set(split_words(normalize_text(b)))
        if not aa or not bb:
            return 0.0
        inter = len(aa.intersection(bb))
        union = len(aa.union(bb))
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=int(row["id"]),
            content=str(row["content"]),
            category=str(row["category"]),
            created_at=str(row["created_at"]),
            use_count=int(row["use_count"]),
        )

    @staticmethod
    def _row_to_learning_item(row: sqlite3.Row) -> AutonomyLearningItem:
        return AutonomyLearningItem(
            id=int(row["id"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            source=str(row["source"]),
            confidence=float(row["confidence"]),
            topic=str(row["topic"]),
            last_checked=str(row["last_checked"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            use_count=int(row["use_count"]),
        )

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    normalized TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT 'general',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    key_name TEXT
                )
                """
            )

            columns = {
                str(row["name"]): str(row["name"]) for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
            }

            if "normalized" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN normalized TEXT NOT NULL DEFAULT ''")
            if "category" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN category TEXT NOT NULL DEFAULT 'general'")
            if "created_at" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN created_at TEXT NOT NULL DEFAULT ''")
            if "last_used_at" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN last_used_at TEXT")
            if "use_count" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN use_count INTEGER NOT NULL DEFAULT 0")
            if "key_name" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN key_name TEXT")

            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_normalized ON memories(normalized)")
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_category_key "
                "ON memories(category, key_name) WHERE key_name IS NOT NULL"
            )

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS autonomy_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload TEXT,
                    reason TEXT NOT NULL DEFAULT '',
                    utility REAL NOT NULL DEFAULT 0.0,
                    exploration REAL NOT NULL DEFAULT 0.0,
                    drives_json TEXT NOT NULL DEFAULT '{}',
                    context_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomy_decisions_created ON autonomy_decisions(created_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomy_decisions_action ON autonomy_decisions(action)"
            )

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS autonomy_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    reward REAL NOT NULL DEFAULT 0.0,
                    message TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(decision_id) REFERENCES autonomy_decisions(id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomy_outcomes_decision ON autonomy_outcomes(decision_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomy_outcomes_created ON autonomy_outcomes(created_at)"
            )

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS autonomy_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    answer TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    topic TEXT NOT NULL DEFAULT '',
                    last_checked TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    use_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_autonomy_learning_question "
                "ON autonomy_learning(normalized_question)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomy_learning_topic ON autonomy_learning(topic)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomy_learning_checked ON autonomy_learning(last_checked)"
            )

    def _backfill_metadata(self) -> None:
        rows = self._conn.execute("SELECT id, content, normalized, key_name FROM memories").fetchall()
        updates: list[tuple[str, str | None, int]] = []
        for row in rows:
            content = str(row["content"])
            normalized = normalize_text(content)
            key_name = self._key_from_content(content)
            existing_norm = str(row["normalized"] or "")
            existing_key = row["key_name"] if row["key_name"] is not None else None
            if existing_norm != normalized or existing_key != key_name:
                updates.append((normalized, key_name, int(row["id"])))

        if not updates:
            return

        with self._conn:
            self._conn.executemany(
                "UPDATE memories SET normalized = ?, key_name = ? WHERE id = ?",
                updates,
            )

    def _maybe_import_legacy_tsv(self) -> None:
        if self.count() > 0:
            return
        if self._legacy_tsv_path is None or not self._legacy_tsv_path.exists():
            return

        try:
            lines = self._legacy_tsv_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return

        imported: list[tuple[int, str, str, str, str, int, str | None]] = []
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

            created_at = self._unescape(parts[2])
            category = self._unescape(parts[3])
            content = self._unescape(parts[4])
            normalized = normalize_text(content)
            key_name = self._key_from_content(content)
            imported.append((item_id, content, normalized, category, created_at, use_count, key_name))

        if not imported:
            return

        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO memories (id, content, normalized, category, created_at, last_used_at, use_count, key_name)
                VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                imported,
            )

            max_id_row = self._conn.execute("SELECT COALESCE(MAX(id), 0) AS mx FROM memories").fetchone()
            max_id = int(max_id_row["mx"]) if max_id_row is not None else 0
            seq_row = self._conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'memories'").fetchone()
            if seq_row is None:
                self._conn.execute("INSERT INTO sqlite_sequence(name, seq) VALUES('memories', ?)", (max_id,))
            else:
                self._conn.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'memories'", (max_id,))

    def add(self, content: str, category: str = "general") -> bool:
        cleaned = trim(content)
        if not cleaned:
            return False
        normalized_new = normalize_text(cleaned)
        if not normalized_new:
            return False

        exists = self._conn.execute("SELECT 1 FROM memories WHERE normalized = ? LIMIT 1", (normalized_new,)).fetchone()
        if exists is not None:
            return False

        now = now_iso8601()
        with self._conn:
            self._conn.execute(
                "INSERT INTO memories(content, normalized, category, created_at, last_used_at, use_count, key_name) VALUES(?, ?, ?, ?, NULL, 0, NULL)",
                (cleaned, normalized_new, category, now),
            )
        return True

    def add_unique(self, content: str, category: str = "general", similarity_threshold: float = 0.88) -> bool:
        cleaned = trim(content)
        if not cleaned:
            return False
        normalized_new = normalize_text(cleaned)
        if not normalized_new:
            return False

        rows = self._conn.execute(
            "SELECT content, normalized FROM memories WHERE category = ?",
            (category,),
        ).fetchall()
        for row in rows:
            existing_content = str(row["content"])
            normalized_existing = str(row["normalized"] or "")
            if not normalized_existing:
                normalized_existing = normalize_text(existing_content)
            if normalized_existing == normalized_new:
                return False
            if normalized_new in normalized_existing or normalized_existing in normalized_new:
                return False
            if self._token_similarity(cleaned, existing_content) >= similarity_threshold:
                return False

        now = now_iso8601()
        with self._conn:
            self._conn.execute(
                "INSERT INTO memories(content, normalized, category, created_at, last_used_at, use_count, key_name) VALUES(?, ?, ?, ?, NULL, 0, NULL)",
                (cleaned, normalized_new, category, now),
            )
        return True

    def upsert(self, key: str, value: str, category: str = "profile") -> bool:
        key_clean = self._normalize_key(key)
        value_clean = trim(value)
        if not key_clean or not value_clean:
            return False

        packed = self._structured_content(key_clean, value_clean)
        normalized_new_value = normalize_text(value_clean)
        now = now_iso8601()

        row = self._conn.execute(
            "SELECT id, content FROM memories WHERE category = ? AND key_name = ? LIMIT 1",
            (category, key_clean),
        ).fetchone()

        if row is not None:
            existing_content = str(row["content"])
            parsed = self._parse_structured(existing_content)
            existing_value = parsed[1] if parsed is not None else existing_content
            if normalize_text(existing_value) == normalized_new_value:
                return False

            with self._conn:
                self._conn.execute(
                    """
                    UPDATE memories
                    SET content = ?, normalized = ?, created_at = ?, last_used_at = NULL, use_count = 0, key_name = ?
                    WHERE id = ?
                    """,
                    (packed, normalize_text(packed), now, key_clean, int(row["id"])),
                )
            return True

        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO memories(content, normalized, category, created_at, last_used_at, use_count, key_name) VALUES(?, ?, ?, ?, NULL, 0, ?)",
                    (packed, normalize_text(packed), category, now, key_clean),
                )
        except sqlite3.IntegrityError:
            # Race-safe fallback: another process inserted the same key concurrently.
            row = self._conn.execute(
                "SELECT id FROM memories WHERE category = ? AND key_name = ? LIMIT 1",
                (category, key_clean),
            ).fetchone()
            if row is None:
                return False
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE memories
                    SET content = ?, normalized = ?, created_at = ?, last_used_at = NULL, use_count = 0, key_name = ?
                    WHERE id = ?
                    """,
                    (packed, normalize_text(packed), now, key_clean, int(row["id"])),
                )
        return True

    def get(self, key: str, category: str = "profile") -> str | None:
        key_clean = self._normalize_key(key)
        if not key_clean:
            return None

        row = self._conn.execute(
            "SELECT content FROM memories WHERE category = ? AND key_name = ? LIMIT 1",
            (category, key_clean),
        ).fetchone()
        if row is None:
            return None

        content = str(row["content"])
        parsed = self._parse_structured(content)
        if parsed is not None:
            return parsed[1]
        return trim(content)

    def log_autonomy_decision(
        self,
        *,
        action: str,
        payload: str | None,
        reason: str,
        utility: float,
        exploration: float,
        drives: dict[str, float] | None = None,
        context: dict[str, object] | None = None,
    ) -> int:
        now = now_iso8601()
        payload_clean = trim(payload or "")
        reason_clean = trim(reason)[:240]
        drives_json = json.dumps(drives or {}, ensure_ascii=True, sort_keys=True)[:4000]
        context_json = json.dumps(context or {}, ensure_ascii=True, sort_keys=True)[:4000]

        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO autonomy_decisions(
                    created_at, action, payload, reason, utility, exploration, drives_json, context_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    trim(action)[:64] or "unknown",
                    payload_clean if payload_clean else None,
                    reason_clean,
                    float(utility),
                    float(exploration),
                    drives_json,
                    context_json,
                ),
            )
        return int(cur.lastrowid) if cur.lastrowid is not None else 0

    def log_autonomy_outcome(
        self,
        *,
        decision_id: int,
        ok: bool,
        reward: float,
        message: str,
        metrics: dict[str, object] | None = None,
    ) -> int:
        if decision_id <= 0:
            return 0

        now = now_iso8601()
        metrics_json = json.dumps(metrics or {}, ensure_ascii=True, sort_keys=True)[:4000]

        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO autonomy_outcomes(
                    decision_id, created_at, ok, reward, message, metrics_json
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    int(decision_id),
                    now,
                    1 if ok else 0,
                    float(reward),
                    trim(message)[:600],
                    metrics_json,
                ),
            )
        return int(cur.lastrowid) if cur.lastrowid is not None else 0

    def autonomy_recent_average_reward(self, limit: int = 20) -> float:
        n = max(1, int(limit))
        row = self._conn.execute(
            "SELECT AVG(reward) AS avg_reward FROM (SELECT reward FROM autonomy_outcomes ORDER BY id DESC LIMIT ?)",
            (n,),
        ).fetchone()
        if row is None:
            return 0.0
        value = row["avg_reward"]
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def upsert_autonomy_learning(
        self,
        *,
        question: str,
        answer: str,
        source: str,
        confidence: float,
        topic: str = "",
        last_checked: str | None = None,
    ) -> bool:
        question_clean = trim(question)
        answer_clean = trim(answer)
        if not question_clean or not answer_clean:
            return False

        normalized_question = normalize_text(question_clean)
        if not normalized_question:
            return False

        source_clean = trim(source) or "unknown"
        topic_clean = trim(topic)
        checked_at = trim(last_checked or "") or now_iso8601()
        now = now_iso8601()
        confidence_value = max(0.0, min(1.0, float(confidence)))

        row = self._conn.execute(
            "SELECT id, answer, source, confidence, topic, last_checked FROM autonomy_learning "
            "WHERE normalized_question = ? LIMIT 1",
            (normalized_question,),
        ).fetchone()

        if row is not None:
            same_answer = normalize_text(str(row["answer"])) == normalize_text(answer_clean)
            same_source = trim(str(row["source"])) == source_clean
            same_topic = trim(str(row["topic"])) == topic_clean
            try:
                same_confidence = abs(float(row["confidence"]) - confidence_value) < 0.0001
            except (TypeError, ValueError):
                same_confidence = False
            same_checked = trim(str(row["last_checked"])) == checked_at
            if same_answer and same_source and same_topic and same_confidence and same_checked:
                return False

            with self._conn:
                self._conn.execute(
                    """
                    UPDATE autonomy_learning
                    SET question = ?, answer = ?, source = ?, confidence = ?, topic = ?,
                        last_checked = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        question_clean,
                        answer_clean,
                        source_clean,
                        confidence_value,
                        topic_clean,
                        checked_at,
                        now,
                        int(row["id"]),
                    ),
                )
            return True

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO autonomy_learning(
                    question, normalized_question, answer, source, confidence, topic,
                    last_checked, created_at, updated_at, use_count
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    question_clean,
                    normalized_question,
                    answer_clean,
                    source_clean,
                    confidence_value,
                    topic_clean,
                    checked_at,
                    now,
                    now,
                ),
            )
        return True

    def get_autonomy_learning(self, question: str) -> AutonomyLearningItem | None:
        normalized_question = normalize_text(question)
        if not normalized_question:
            return None
        row = self._conn.execute(
            "SELECT * FROM autonomy_learning WHERE normalized_question = ? LIMIT 1",
            (normalized_question,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_learning_item(row)

    def recent_autonomy_learning(self, limit: int = 10) -> list[AutonomyLearningItem]:
        n = max(1, int(limit))
        rows = self._conn.execute(
            "SELECT * FROM autonomy_learning ORDER BY updated_at DESC, id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [self._row_to_learning_item(row) for row in rows]

    def bump_counter(self, key: str, delta: int = 1, category: str = "stats") -> int:
        key_clean = self._normalize_key(key)
        if not key_clean:
            return 0

        row = self._conn.execute(
            "SELECT id, content FROM memories WHERE category = ? AND key_name = ? LIMIT 1",
            (category, key_clean),
        ).fetchone()

        current = 0
        row_id: int | None = None
        if row is not None:
            row_id = int(row["id"])
            parsed = self._parse_structured(str(row["content"]))
            raw_value = parsed[1] if parsed is not None else str(row["content"])
            m = re.search(r"-?\d+", raw_value)
            if m:
                try:
                    current = int(m.group(0))
                except ValueError:
                    current = 0

        new_value = current + int(delta)
        if new_value < 0:
            new_value = 0

        packed = self._structured_content(key_clean, str(new_value))
        normalized = normalize_text(packed)
        now = now_iso8601()

        with self._conn:
            if row_id is not None:
                self._conn.execute(
                    """
                    UPDATE memories
                    SET content = ?, normalized = ?, created_at = ?, last_used_at = NULL, use_count = 0, key_name = ?
                    WHERE id = ?
                    """,
                    (packed, normalized, now, key_clean, row_id),
                )
            else:
                self._conn.execute(
                    "INSERT INTO memories(content, normalized, category, created_at, last_used_at, use_count, key_name) VALUES(?, ?, ?, ?, NULL, 0, ?)",
                    (packed, normalized, category, now, key_clean),
                )

        return new_value

    def recent(self, limit: int = 5) -> list[MemoryItem]:
        if limit <= 0:
            limit = 5
        rows = self._conn.execute(
            "SELECT id, content, category, created_at, use_count FROM memories ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        if limit <= 0:
            limit = 5

        q = normalize_text(query)
        if not q:
            return self.recent(limit)

        q_tokens = split_words(q)
        rows = self._conn.execute(
            "SELECT id, content, category, created_at, use_count, normalized FROM memories"
        ).fetchall()

        scored: list[tuple[int, sqlite3.Row]] = []
        for row in rows:
            normalized = str(row["normalized"] or "")
            if not normalized:
                normalized = normalize_text(str(row["content"]))

            score = 0
            if q in normalized:
                score += 8
            for token in q_tokens:
                if len(token) < 3:
                    continue
                if token in normalized:
                    score += 2
            score += min(int(row["use_count"]), 5)
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        selected_rows = [row for _, row in scored[:limit]]
        if not selected_rows:
            return []

        now = now_iso8601()
        ids = [int(row["id"]) for row in selected_rows]
        with self._conn:
            self._conn.executemany(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                [(now, item_id) for item_id in ids],
            )

        result: list[MemoryItem] = []
        for row in selected_rows:
            item = self._row_to_item(row)
            item.use_count += 1
            result.append(item)
        return result
