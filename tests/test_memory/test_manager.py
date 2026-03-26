"""Tests for MemoryManager."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ss.blackboard.database import Database
from ss.blackboard.models import AgentName, Memory, MemoryScope, MemoryType
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.memory.manager import MemoryManager
from ss.vectors.encoder import EmbeddingEncoder
from ss.vectors.store import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(config: Config):
    database = Database()
    database.connect(config)
    yield database
    database.close()


@pytest.fixture
def repo(db):
    return Repository(db)


@pytest.fixture
def encoder():
    return EmbeddingEncoder()


@pytest.fixture
def vector_store(db, encoder):
    return VectorStore(db, encoder)


@pytest.fixture
def manager(repo, vector_store, config):
    return MemoryManager(repo, vector_store, config)


def _make_memory(
    content: str = "Test memory content",
    type: MemoryType = MemoryType.knowledge,
    scope: MemoryScope = MemoryScope.long_term,
    confidence: float = 1.0,
    is_active: bool = True,
) -> Memory:
    return Memory(
        content=content,
        type=type,
        scope=scope,
        source_agent=AgentName.reception,
        confidence=confidence,
        is_active=is_active,
    )


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------

def test_store_returns_memory_id(manager):
    memory = _make_memory("Python is great for scripting.")
    result = manager.store(memory)
    assert result == memory.id


def test_store_persists_memory(manager, repo):
    memory = _make_memory("Stored memory content.")
    manager.store(memory)

    fetched = repo.get_memory(memory.id)
    assert fetched is not None
    assert fetched.content == "Stored memory content."


def test_store_indexes_for_search(manager, vector_store):
    memory = _make_memory("Machine learning improves predictions.")
    manager.store(memory)

    results = vector_store.search("memory", "machine learning predictions", limit=5)
    entity_ids = [r["entity_id"] for r in results]
    assert memory.id in entity_ids


# ---------------------------------------------------------------------------
# Recall tests
# ---------------------------------------------------------------------------

def test_recall_returns_memories(manager):
    manager.store(_make_memory("Python async programming patterns."))
    manager.store(_make_memory("JavaScript async programming patterns."))

    results = manager.recall("async programming", limit=5)
    assert len(results) >= 1


def test_recall_filters_by_type(manager):
    manager.store(_make_memory("Good pattern: use dependency injection.", type=MemoryType.good_practice))
    manager.store(_make_memory("Anti-pattern: global state.", type=MemoryType.anti_pattern))

    results = manager.recall("pattern", type=MemoryType.good_practice, limit=5)
    for m in results:
        assert m.type == MemoryType.good_practice


def test_recall_filters_by_scope(manager):
    manager.store(_make_memory("Short-term note.", scope=MemoryScope.short_term))
    manager.store(_make_memory("Long-term knowledge.", scope=MemoryScope.long_term))

    results = manager.recall("note knowledge", scope=MemoryScope.long_term, limit=5)
    for m in results:
        assert m.scope == MemoryScope.long_term


def test_recall_filters_below_min_confidence(manager, repo):
    memory = _make_memory("Low confidence memory.")
    manager.store(memory)
    repo.update_memory_confidence(memory.id, 0.1)

    results = manager.recall("low confidence memory", min_confidence=0.3, limit=5)
    ids = [m.id for m in results]
    assert memory.id not in ids


def test_recall_excludes_inactive_memories(manager, repo):
    memory = _make_memory("Inactive memory content.")
    manager.store(memory)
    repo.supersede_memory(memory.id, "some-other-id")

    results = manager.recall("inactive memory", limit=5)
    ids = [m.id for m in results]
    assert memory.id not in ids


def test_recall_increments_relevance_count(manager, repo):
    memory = _make_memory("Relevance count test memory.")
    manager.store(memory)

    # Recall should increment relevance_count
    manager.recall("relevance count test memory", limit=5)

    updated = repo.get_memory(memory.id)
    assert updated is not None
    assert updated.relevance_count == 1


def test_recall_increments_relevance_on_multiple_recalls(manager, repo):
    memory = _make_memory("Multiple recall test memory.")
    manager.store(memory)

    manager.recall("multiple recall test memory", limit=5)
    manager.recall("multiple recall test memory", limit=5)

    updated = repo.get_memory(memory.id)
    assert updated is not None
    assert updated.relevance_count == 2


def test_recall_returns_empty_when_no_memories(manager):
    results = manager.recall("nothing here", limit=5)
    assert results == []


def test_recall_respects_limit(manager):
    for i in range(5):
        manager.store(_make_memory(f"Memory about coding practice number {i}."))

    results = manager.recall("coding practice", limit=2)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

def test_store_supersedes_near_duplicate(manager, repo, config):
    # Store original memory
    original = _make_memory("Use try/except for error handling in Python.")
    manager.store(original)

    # Store nearly identical memory — distance should be below threshold
    # Use a very similar text to trigger the near-duplicate path
    duplicate = _make_memory("Use try/except for error handling in Python.")
    duplicate_id = duplicate.id
    # Make duplicate a new object with same content (same hash → distance ~0)
    manager.store(duplicate)

    # The original should be superseded
    updated_original = repo.get_memory(original.id)
    assert updated_original is not None
    assert updated_original.is_active is False
    assert updated_original.superseded_by == duplicate.id
