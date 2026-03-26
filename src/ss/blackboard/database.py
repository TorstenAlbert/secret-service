"""SQLite database wrapper for the SS blackboard."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vss

from ss.config import Config


_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """Wraps a SQLite connection, loads extensions, and runs schema migrations."""

    def __init__(self) -> None:
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self, config: Config) -> None:
        """Open (or create) the database at config.db_path and run migrations."""
        db_path = Path(config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Enable extension loading before loading sqlite-vss
        self._conn.enable_load_extension(True)
        sqlite_vss.load(self._conn)

        # Performance / reliability pragmas
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")

        self._migrate()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        """Execute the schema DDL (idempotent — all statements use IF NOT EXISTS)."""
        schema = _SCHEMA_PATH.read_text()
        self._conn.executescript(schema)  # type: ignore[union-attr]
