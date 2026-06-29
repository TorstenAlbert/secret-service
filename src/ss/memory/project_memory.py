"""Project Memory Layer (PML): local persistent project-intelligence store.

Records WHY a project is the way it is across sessions (modelled on the
project-intelligence idea: capture/inspect verbs over STATE/INTENT/DECISION/
WHY/TIMELINE/HEALTH dimensions, stored locally). Deterministic, LLM-free.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

RECORD_TYPES = ("STATE", "INTENT", "DECISION", "WHY", "TIMELINE", "HEALTH")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectMemory:
    """Local-first project intelligence store under ``root`` (one SQLite file)."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._root / "pml.db"), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                record_type TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                refs TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_records_type ON records(record_type, created_at)"
        )
        self._conn.commit()

    # -- writes -------------------------------------------------------------
    def capture(
        self, record_type: str, content: str, *,
        tags: list[str] | None = None, refs: dict | None = None,
    ) -> str:
        if record_type not in RECORD_TYPES:
            raise ValueError(f"unknown record_type {record_type!r}; expected one of {RECORD_TYPES}")
        rid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO records (id, record_type, content, tags, refs, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rid, record_type, content, json.dumps(tags or []),
             json.dumps(refs or {}), _now_iso()),
        )
        self._conn.commit()
        return rid

    def capture_decision(self, content: str, **kw) -> str:
        return self.capture("DECISION", content, **kw)

    def capture_why(self, content: str, **kw) -> str:
        return self.capture("WHY", content, **kw)

    def capture_timeline(self, content: str, **kw) -> str:
        return self.capture("TIMELINE", content, **kw)

    def capture_note(self, content: str, **kw) -> str:
        # A note is a STATE observation about how things stand.
        return self.capture("STATE", content, **kw)

    # -- reads --------------------------------------------------------------
    def inspect(self, record_type: str, *, limit: int = 10) -> list[dict]:
        cur = self._conn.execute(
            "SELECT id, record_type, content, tags, refs, created_at FROM records "
            "WHERE record_type = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (record_type, limit),
        )
        out = []
        for row in cur.fetchall():
            out.append({
                "id": row["id"], "record_type": row["record_type"],
                "content": row["content"], "tags": json.loads(row["tags"] or "[]"),
                "refs": json.loads(row["refs"] or "{}"), "created_at": row["created_at"],
            })
        return out

    def state(self, limit: int = 10) -> list[dict]:
        return self.inspect("STATE", limit=limit)

    def health(self, limit: int = 10) -> list[dict]:
        return self.inspect("HEALTH", limit=limit)

    def decisions(self, limit: int = 10) -> list[dict]:
        return self.inspect("DECISION", limit=limit)

    def intent(self, limit: int = 10) -> list[dict]:
        return self.inspect("INTENT", limit=limit)

    def timeline(self, limit: int = 10) -> list[dict]:
        return self.inspect("TIMELINE", limit=limit)

    def as_context(self, record_types: list[str], *, per_type: int = 5) -> str:
        blocks: list[str] = []
        for rt in record_types:
            rows = self.inspect(rt, limit=per_type)
            if not rows:
                continue
            lines = "\n".join(f"- {r['content']}" for r in rows)
            blocks.append(f"## PROJECT {rt}\n{lines}")
        return "\n\n".join(blocks)

    def close(self) -> None:
        self._conn.close()
