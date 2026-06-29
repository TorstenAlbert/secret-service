"""Code Index Layer (CIL): token-efficient code navigation + live log access.

Indexes a Python source tree once (via ``ast``) into a local SQLite database,
then answers identifier-level queries, signatures, change-detection, notes,
tasks, and reads a local live-log file — modelled on persistent code-indexing
for AI assistants. CIL lowers token cost PER call; it does not change the number
of LLM calls (that is the loop budget's job).
"""
from __future__ import annotations

import ast
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [a.arg for a in node.args.args]
        return f"{node.name}({', '.join(args)})"
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    return ""


class CodeIndex:
    """Local persistent identifier/signature index + notes/tasks + log reader."""

    def __init__(self, root_dir, store_dir=".code-index", encoder=None, log_file=None) -> None:
        self._root = Path(root_dir)
        self._store = Path(store_dir)
        self._store.mkdir(parents=True, exist_ok=True)
        self._encoder = encoder
        self._log_file = Path(log_file) if log_file else (self._store / "live.log")
        self._conn = sqlite3.connect(str(self._store / "index.db"), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self) -> None:
        c = self._conn
        c.execute("""CREATE TABLE IF NOT EXISTS identifiers (
            id TEXT PRIMARY KEY, name TEXT, kind TEXT, signature TEXT,
            file TEXT, line INTEGER)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ident_name ON identifiers(name)")
        c.execute("""CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY, hash TEXT, mtime REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY, path TEXT, note TEXT, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, title TEXT, status TEXT, priority TEXT,
            tags TEXT, created_at TEXT)""")
        c.commit()

    # -- indexing -----------------------------------------------------------
    def _index_file(self, path: Path) -> int:
        rel = str(path.relative_to(self._root))
        self._conn.execute("DELETE FROM identifiers WHERE file = ?", (rel,))
        count = 0
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    self._conn.execute(
                        "INSERT INTO identifiers (id, name, kind, signature, file, line) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (str(uuid.uuid4()), node.name, kind, _signature(node), rel, node.lineno),
                    )
                    count += 1
        st = path.stat()
        self._conn.execute(
            "INSERT OR REPLACE INTO files (path, hash, mtime) VALUES (?, ?, ?)",
            (rel, _file_hash(path), st.st_mtime),
        )
        self._conn.commit()
        return count

    def index(self) -> int:
        total = 0
        for path in sorted(self._root.rglob("*.py")):
            total += self._index_file(path)
        return total

    # -- queries ------------------------------------------------------------
    def summary(self) -> dict:
        c = self._conn
        files = c.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        idents = c.execute("SELECT COUNT(*) FROM identifiers").fetchone()[0]
        by_kind = {row["kind"]: row["n"] for row in c.execute(
            "SELECT kind, COUNT(*) n FROM identifiers GROUP BY kind")}
        return {"files": files, "identifiers": idents, "by_kind": by_kind,
                "languages": {"python": files}}

    def signatures(self, path: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT name, kind, signature, file, line FROM identifiers "
            "WHERE file = ? OR file LIKE ? ORDER BY line", (path, f"{path}%")).fetchall()
        return [dict(r) for r in rows]

    def query(self, term: str, *, mode: str = "contains",
              modified_since: float | None = None, limit: int = 20) -> list[dict]:
        if mode == "exact":
            where, param = "name = ?", term
        elif mode == "starts_with":
            where, param = "name LIKE ?", f"{term}%"
        else:
            where, param = "name LIKE ?", f"%{term}%"
        rows = self._conn.execute(
            f"SELECT name, kind, file, line FROM identifiers WHERE {where} LIMIT ?",
            (param, limit)).fetchall()
        results = [dict(r) for r in rows]
        if modified_since is not None:
            mtimes = {r["path"]: r["mtime"] for r in self._conn.execute(
                "SELECT path, mtime FROM files").fetchall()}
            results = [r for r in results if mtimes.get(r["file"], 0) >= modified_since]
        return results

    def semantic_query(self, text: str, *, limit: int = 10) -> list[dict]:
        if self._encoder is None:
            return self.query(text, mode="contains", limit=limit)
        import numpy as np
        q = self._encoder.encode(text)
        scored = []
        for r in self._conn.execute(
                "SELECT name, kind, signature, file, line FROM identifiers").fetchall():
            v = self._encoder.encode(f"{r['name']} {r['signature']}")
            denom = float(np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
            scored.append((float(np.dot(q, v)) / denom, dict(r)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def session(self, path: str | None = None) -> dict:
        changed, reindexed = [], 0
        known = {r["path"]: r["hash"] for r in self._conn.execute(
            "SELECT path, hash FROM files").fetchall()}
        targets = ([self._root / path] if path else list(self._root.rglob("*.py")))
        for p in targets:
            if not p.exists():
                continue
            rel = str(p.relative_to(self._root))
            if known.get(rel) != _file_hash(p):
                self._index_file(p)
                changed.append(rel)
                reindexed += 1
        return {"changed": changed, "reindexed": reindexed}

    # -- notes / tasks ------------------------------------------------------
    def note(self, path: str, note: str | None = None):
        if note is not None:
            nid = str(uuid.uuid4())
            self._conn.execute(
                "INSERT INTO notes (id, path, note, created_at) VALUES (?, ?, ?, ?)",
                (nid, path, note, _now_iso()))
            self._conn.commit()
            return nid
        rows = self._conn.execute(
            "SELECT id, path, note, created_at FROM notes WHERE path = ? "
            "ORDER BY created_at DESC", (path,)).fetchall()
        return [dict(r) for r in rows]

    def task(self, action: str, **kw) -> Any:
        if action == "open":
            tid = str(uuid.uuid4())
            self._conn.execute(
                "INSERT INTO tasks (id, title, status, priority, tags, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tid, kw.get("title", ""), "open", kw.get("priority", "medium"),
                 json.dumps(kw.get("tags", [])), _now_iso()))
            self._conn.commit()
            return {"id": tid, "status": "open"}
        if action == "update":
            self._conn.execute("UPDATE tasks SET status = ? WHERE id = ?",
                               (kw.get("status", "open"), kw.get("id")))
            self._conn.commit()
            return {"id": kw.get("id"), "status": kw.get("status")}
        rows = self._conn.execute(
            "SELECT id, title, status, priority, tags, created_at FROM tasks "
            "ORDER BY created_at DESC").fetchall()
        return [{**dict(r), "tags": json.loads(r["tags"] or "[]")} for r in rows]

    # -- live log (local file, no HTTP) -------------------------------------
    def log(self, action: str = "query", *, level: str | None = None,
            since: float | None = None, limit: int = 100) -> list[dict]:
        if not self._log_file.exists():
            return []
        out = []
        for line in self._log_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if level and entry.get("level") != level:
                continue
            if since is not None and float(entry.get("ts", 0)) < since:
                continue
            out.append(entry)
        return out[-limit:]

    def summary_text(self) -> str:
        s = self.summary()
        return (f"Code index: {s['files']} files, {s['identifiers']} identifiers "
                f"({s.get('by_kind', {})}).")

    def close(self) -> None:
        self._conn.close()
