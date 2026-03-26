"""Tests for the Database class."""
from pathlib import Path

import pytest

from ss.blackboard.database import Database
from ss.config import Config


EXPECTED_TABLES = {
    "sessions",
    "issues",
    "strategies",
    "taktiks",
    "missions",
    "mission_results",
    "memories",
    "agent_notes",
    "client_profiles",
    "client_issue_history",
    "strategy_scores",
    "session_events",
    "embedding_registry",
}

VSS_TABLES = {
    "vss_issues",
    "vss_strategies",
    "vss_taktiks",
    "vss_missions",
    "vss_memories",
    "vss_agent_notes",
}


@pytest.fixture
def db(config: Config) -> Database:
    database = Database()
    database.connect(config)
    yield database
    database.close()


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

def test_db_file_created(config: Config, tmp_path: Path):
    database = Database()
    database.connect(config)
    database.close()
    assert config.db_path.exists()


def test_parent_dirs_created(tmp_path: Path):
    nested_path = tmp_path / "a" / "b" / "c" / "test.db"
    cfg = Config(db_path=nested_path)
    database = Database()
    database.connect(cfg)
    database.close()
    assert nested_path.exists()


# ---------------------------------------------------------------------------
# PRAGMA verification
# ---------------------------------------------------------------------------

def test_wal_mode(db: Database):
    row = db.conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0] == "wal"


def test_foreign_keys_on(db: Database):
    row = db.conn.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1


# ---------------------------------------------------------------------------
# Table existence
# ---------------------------------------------------------------------------

def _get_tables(db: Database) -> set[str]:
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {row[0] for row in rows}


def test_all_regular_tables_exist(db: Database):
    tables = _get_tables(db)
    for table in EXPECTED_TABLES:
        assert table in tables, f"Missing table: {table}"


def test_vss_tables_exist(db: Database):
    # vss virtual tables show up in sqlite_master with type='table' or 'shadow'
    rows = db.conn.execute("SELECT name FROM sqlite_master").fetchall()
    names = {row[0] for row in rows}
    for vss_table in VSS_TABLES:
        assert vss_table in names, f"Missing vss table: {vss_table}"


# ---------------------------------------------------------------------------
# Idempotent migration
# ---------------------------------------------------------------------------

def test_idempotent_migration(config: Config):
    """Calling connect twice (re-running schema) should not raise."""
    db = Database()
    db.connect(config)
    db._migrate()  # Run again — must not fail
    db.close()


# ---------------------------------------------------------------------------
# Connection guard
# ---------------------------------------------------------------------------

def test_conn_raises_before_connect():
    db = Database()
    with pytest.raises(RuntimeError, match="not connected"):
        _ = db.conn
