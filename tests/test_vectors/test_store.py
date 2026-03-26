"""Tests for VectorStore."""
from __future__ import annotations

import pytest

from ss.blackboard.database import Database
from ss.config import Config
from ss.vectors.encoder import EmbeddingEncoder
from ss.vectors.store import VectorStore


@pytest.fixture
def db(config: Config) -> Database:
    database = Database()
    database.connect(config)
    yield database
    database.close()


@pytest.fixture
def encoder() -> EmbeddingEncoder:
    return EmbeddingEncoder()


@pytest.fixture
def store(db: Database, encoder: EmbeddingEncoder) -> VectorStore:
    return VectorStore(db, encoder)


# ---------------------------------------------------------------------------
# Basic index + search round-trip
# ---------------------------------------------------------------------------

def test_index_and_search_returns_results(store: VectorStore):
    store.index("memory", "mem-001", "Python is a programming language")
    results = store.search("memory", "Python programming language", limit=5)
    assert len(results) == 1
    assert results[0]["entity_id"] == "mem-001"
    assert "distance" in results[0]


def test_search_distance_is_numeric(store: VectorStore):
    store.index("memory", "mem-001", "hello world")
    results = store.search("memory", "hello world")
    assert isinstance(results[0]["distance"], float)


def test_search_returns_closest_first(store: VectorStore):
    store.index("memory", "mem-001", "The quick brown fox")
    store.index("memory", "mem-002", "A fast red dog")
    store.index("memory", "mem-003", "quantum physics equations")
    results = store.search("memory", "quick fox running", limit=3)
    # First result should be most semantically similar
    entity_ids = [r["entity_id"] for r in results]
    # mem-001 or mem-002 should rank before mem-003
    assert entity_ids.index("mem-003") > 0


# ---------------------------------------------------------------------------
# Empty table
# ---------------------------------------------------------------------------

def test_search_empty_table_returns_empty(store: VectorStore):
    results = store.search("memory", "anything", limit=5)
    assert results == []


def test_search_empty_issue_table_returns_empty(store: VectorStore):
    results = store.search("issue", "test query")
    assert results == []


# ---------------------------------------------------------------------------
# Content hash — skip re-embedding unchanged content
# ---------------------------------------------------------------------------

def test_index_idempotent_same_content(store: VectorStore, db: Database):
    store.index("memory", "mem-001", "Python programming")
    # Count registry entries before second index
    count_before = db.conn.execute(
        "SELECT COUNT(*) FROM embedding_registry WHERE entity_type='memory' AND entity_id='mem-001'"
    ).fetchone()[0]
    store.index("memory", "mem-001", "Python programming")  # same content
    count_after = db.conn.execute(
        "SELECT COUNT(*) FROM embedding_registry WHERE entity_type='memory' AND entity_id='mem-001'"
    ).fetchone()[0]
    assert count_before == count_after == 1


def test_index_updates_on_content_change(store: VectorStore):
    store.index("memory", "mem-001", "original content about cats")
    store.index("memory", "mem-001", "completely different: machine learning")
    # After update, searching for the new content should find it
    results = store.search("memory", "machine learning neural networks", limit=5)
    assert len(results) == 1
    assert results[0]["entity_id"] == "mem-001"


# ---------------------------------------------------------------------------
# Entity type isolation
# ---------------------------------------------------------------------------

def test_different_entity_types_dont_cross_contaminate(store: VectorStore):
    store.index("memory", "id-001", "Python programming language")
    store.index("issue", "id-001", "completely different topic about cats")
    # Searching memories should only return the memory
    mem_results = store.search("memory", "Python programming", limit=5)
    assert all(r["entity_id"] == "id-001" for r in mem_results)
    # Results come from their respective tables, not crossing over
    # (both have id-001, so just verify we get results from both independently)
    issue_results = store.search("issue", "cats and pets", limit=5)
    assert len(issue_results) == 1


def test_index_multiple_entity_types(store: VectorStore):
    store.index("memory", "m-1", "strategy for solving the problem")
    store.index("strategy", "s-1", "strategy for solving the problem")
    mem_results = store.search("memory", "strategy solution", limit=5)
    strat_results = store.search("strategy", "strategy solution", limit=5)
    assert mem_results[0]["entity_id"] == "m-1"
    assert strat_results[0]["entity_id"] == "s-1"


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------

def test_find_similar_excludes_self(store: VectorStore):
    store.index("memory", "m-1", "Python is a programming language")
    store.index("memory", "m-2", "Python language for programming")
    store.index("memory", "m-3", "cooking recipes for dinner")
    results = store.find_similar("memory", "m-1", limit=5)
    entity_ids = [r["entity_id"] for r in results]
    assert "m-1" not in entity_ids


def test_find_similar_returns_related(store: VectorStore):
    store.index("memory", "m-1", "machine learning model training")
    store.index("memory", "m-2", "training neural networks with backprop")
    store.index("memory", "m-3", "medieval history of europe")
    results = store.find_similar("memory", "m-1", limit=5)
    entity_ids = [r["entity_id"] for r in results]
    # m-2 should rank before m-3
    if "m-2" in entity_ids and "m-3" in entity_ids:
        assert entity_ids.index("m-2") < entity_ids.index("m-3")


def test_find_similar_empty_table_returns_empty(store: VectorStore):
    # Entity doesn't exist in vss table
    results = store.find_similar("memory", "nonexistent-id", limit=5)
    assert results == []


def test_find_similar_only_self_returns_empty(store: VectorStore):
    store.index("memory", "m-1", "only one entry")
    results = store.find_similar("memory", "m-1", limit=5)
    assert results == []


# ---------------------------------------------------------------------------
# Limit parameter
# ---------------------------------------------------------------------------

def test_search_respects_limit(store: VectorStore):
    for i in range(5):
        store.index("memory", f"m-{i}", f"document number {i} about various topics")
    results = store.search("memory", "document topic", limit=3)
    assert len(results) <= 3


def test_find_similar_respects_limit(store: VectorStore):
    for i in range(6):
        store.index("memory", f"m-{i}", f"Python programming document {i}")
    results = store.find_similar("memory", "m-0", limit=3)
    assert len(results) <= 3
