"""Tests for MemoryCleanup and ClientProfileManager."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ss.blackboard.database import Database
from ss.blackboard.models import AgentName, ClientProfile, Memory, MemoryScope, MemoryType
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.memory.cleanup import MemoryCleanup
from ss.memory.client_profile import ClientProfileManager


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
def cleanup(repo, config):
    return MemoryCleanup(repo, config)


@pytest.fixture
def profile_mgr(repo):
    return ClientProfileManager(repo)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_memory(
    content: str = "Test content",
    scope: MemoryScope = MemoryScope.short_term,
    expires_at: datetime | None = None,
) -> Memory:
    return Memory(
        content=content,
        type=MemoryType.knowledge,
        scope=scope,
        source_agent=AgentName.reception,
        expires_at=expires_at,
    )


# ---------------------------------------------------------------------------
# MemoryCleanup tests
# ---------------------------------------------------------------------------

def test_cleanup_expired_deactivates_expired_memories(cleanup, repo):
    past = _now() - timedelta(hours=1)
    memory = _make_memory("Expired short-term memory.", expires_at=past)
    repo.insert_memory(memory)

    count = cleanup.cleanup_expired()
    assert count >= 1

    updated = repo.get_memory(memory.id)
    assert updated is not None
    assert updated.is_active is False


def test_cleanup_expired_does_not_deactivate_future_expiry(cleanup, repo):
    future = _now() + timedelta(hours=1)
    memory = _make_memory("Still valid memory.", expires_at=future)
    repo.insert_memory(memory)

    cleanup.cleanup_expired()

    updated = repo.get_memory(memory.id)
    assert updated is not None
    assert updated.is_active is True


def test_cleanup_expired_ignores_memories_without_expiry(cleanup, repo):
    memory = _make_memory("Permanent memory.", expires_at=None)
    memory = memory.model_copy(update={"scope": MemoryScope.permanent})
    repo.insert_memory(memory)

    cleanup.cleanup_expired()

    updated = repo.get_memory(memory.id)
    assert updated is not None
    assert updated.is_active is True


def test_cleanup_expired_returns_count(cleanup, repo):
    past = _now() - timedelta(hours=1)
    for i in range(3):
        repo.insert_memory(_make_memory(f"Expired memory {i}", expires_at=past))

    count = cleanup.cleanup_expired()
    assert count == 3


def test_cleanup_expired_returns_zero_when_nothing_to_expire(cleanup):
    count = cleanup.cleanup_expired()
    assert count == 0


def test_decay_confidence_reduces_by_decay_rate(cleanup, repo, config):
    memory = _make_memory("Memory to decay.")
    repo.insert_memory(memory)
    initial_confidence = memory.confidence

    cleanup.decay_confidence(memory.id, "test decay")

    updated = repo.get_memory(memory.id)
    assert updated is not None
    expected = initial_confidence - config.confidence_decay_rate
    assert abs(updated.confidence - expected) < 1e-9


def test_decay_confidence_clamps_at_zero(cleanup, repo, config):
    memory = _make_memory("Almost dead memory.")
    repo.insert_memory(memory)
    # Set confidence to near zero
    repo.update_memory_confidence(memory.id, 0.01)

    cleanup.decay_confidence(memory.id, "decay to zero")

    updated = repo.get_memory(memory.id)
    assert updated is not None
    assert updated.confidence >= 0.0


def test_decay_confidence_on_missing_memory_does_not_raise(cleanup):
    # Should silently do nothing for unknown IDs
    cleanup.decay_confidence("nonexistent-id", "reason")


# ---------------------------------------------------------------------------
# ClientProfileManager tests
# ---------------------------------------------------------------------------

def test_get_or_create_creates_new_profile(profile_mgr):
    profile = profile_mgr.get_or_create("client-abc")
    assert profile.client_id == "client-abc"


def test_get_or_create_returns_existing_profile(profile_mgr):
    profile1 = profile_mgr.get_or_create("client-xyz")
    profile2 = profile_mgr.get_or_create("client-xyz")
    assert profile1.client_id == profile2.client_id


def test_get_or_create_persists_profile(profile_mgr, repo):
    profile_mgr.get_or_create("client-persist")
    fetched = repo.get_client_profile("client-persist")
    assert fetched is not None
    assert fetched.client_id == "client-persist"


def test_update_after_session_increments_total_sessions(profile_mgr):
    profile_mgr.get_or_create("client-sessions")
    updated = profile_mgr.update_after_session("client-sessions")
    assert updated.total_sessions == 1

    updated2 = profile_mgr.update_after_session("client-sessions")
    assert updated2.total_sessions == 2


def test_update_after_session_updates_expertise_level(profile_mgr):
    profile_mgr.get_or_create("client-expertise")
    updated = profile_mgr.update_after_session("client-expertise", expertise_level="senior")
    assert updated.expertise_level == "senior"


def test_update_after_session_merges_known_domains(profile_mgr):
    profile_mgr.get_or_create("client-domains")
    profile_mgr.update_after_session("client-domains", known_domains=["python", "ml"])
    updated = profile_mgr.update_after_session("client-domains", known_domains=["ml", "devops"])

    # "ml" should appear only once; both python and devops should be present
    assert "python" in updated.known_domains
    assert "ml" in updated.known_domains
    assert "devops" in updated.known_domains
    assert updated.known_domains.count("ml") == 1


def test_update_after_session_updates_communication_style(profile_mgr):
    profile_mgr.get_or_create("client-style")
    updated = profile_mgr.update_after_session("client-style", communication_style="concise")
    assert updated.communication_style == "concise"


def test_update_after_session_creates_profile_if_missing(profile_mgr):
    # No get_or_create called first
    updated = profile_mgr.update_after_session("client-new", expertise_level="junior")
    assert updated.client_id == "client-new"
    assert updated.expertise_level == "junior"
    assert updated.total_sessions == 1
