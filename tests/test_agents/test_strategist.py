"""Tests for StrategistAgent."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ss.agents.strategist import StrategistAgent
from ss.blackboard.database import Database
from ss.blackboard.models import AgentName, Issue, IssueClassification, Session
from ss.blackboard.repository import Repository
from ss.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path):
    config = Config(db_path=tmp_path / "test.db")
    database = Database()
    database.connect(config)
    yield database
    database.close()


@pytest.fixture
def repo(db):
    return Repository(db)


@pytest.fixture
def mock_sampling():
    m = MagicMock()
    m.complete = AsyncMock(return_value="text")
    m.complete_structured = AsyncMock(return_value={
        "strategies": [
            {
                "description": "Add connection pooling",
                "objective": "Prevent memory leaks from unclosed connections",
                "approach_type": "refactor",
                "confidence": 0.9,
                "rank": 1,
            },
            {
                "description": "Implement connection timeout",
                "objective": "Ensure connections are released",
                "approach_type": "defensive",
                "confidence": 0.75,
                "rank": 2,
            },
            {
                "description": "Use context managers for all DB operations",
                "objective": "Guarantee connection cleanup",
                "approach_type": "pattern",
                "confidence": 0.85,
                "rank": 3,
            },
        ]
    })
    return m


@pytest.fixture
def mock_memory_mgr():
    m = MagicMock()
    m.recall = MagicMock(return_value=[])
    return m


@pytest.fixture
def mock_vector_store():
    m = MagicMock()
    m.index = MagicMock()
    return m


@pytest.fixture
def agent(repo, mock_sampling, mock_memory_mgr, mock_vector_store):
    return StrategistAgent(repo, mock_sampling, mock_memory_mgr, mock_vector_store)


@pytest.fixture
def session_id(repo):
    s = Session(client_id="client-1", problem_text="test problem")
    repo.insert_session(s)
    return s.id


@pytest.fixture
def issue(repo, session_id):
    i = Issue(
        session_id=session_id,
        summary="Memory leak in user service",
        classification=IssueClassification.bug,
        severity="high",
        who="backend engineers",
        where_location="user-service/memory.py",
        why_reason="Unclosed database connections",
        precondition="Service running under load",
        postcondition="OOM error",
        key_points=["OOM", "memory leak"],
    )
    repo.insert_issue(i)
    return i


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_strategist_agent_properties(agent):
    assert agent.name == AgentName.strategist
    assert agent.temperature == 0.9
    assert "Strategist" in agent.persona


@pytest.mark.asyncio
async def test_execute_returns_strategies(agent, session_id, issue):
    strategies = await agent.execute(session_id, issue=issue, num_strategies=3)
    assert len(strategies) == 3


@pytest.mark.asyncio
async def test_execute_strategies_have_correct_fields(agent, session_id, issue):
    strategies = await agent.execute(session_id, issue=issue)
    for s in strategies:
        assert s.session_id == session_id
        assert s.description
        assert s.objective
        assert s.rank is not None


@pytest.mark.asyncio
async def test_execute_persists_strategies(agent, repo, session_id, issue):
    strategies = await agent.execute(session_id, issue=issue, num_strategies=3)
    db_strategies = repo.list_strategies(session_id)
    assert len(db_strategies) == 3
    db_ids = {s.id for s in db_strategies}
    for s in strategies:
        assert s.id in db_ids


@pytest.mark.asyncio
async def test_execute_indexes_vectors(agent, mock_vector_store, session_id, issue):
    strategies = await agent.execute(session_id, issue=issue, num_strategies=3)
    assert mock_vector_store.index.call_count == 3
    for s in strategies:
        mock_vector_store.index.assert_any_call("strategy", s.id, s.description)


@pytest.mark.asyncio
async def test_execute_emits_strategies_generated_event(agent, repo, session_id, issue):
    await agent.execute(session_id, issue=issue)
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "strategies_generated" in event_types


@pytest.mark.asyncio
async def test_execute_recalls_memories(agent, mock_memory_mgr, session_id, issue):
    await agent.execute(session_id, issue=issue)
    assert mock_memory_mgr.recall.call_count >= 2  # good_practice + bad_practice


@pytest.mark.asyncio
async def test_execute_with_failure_context(agent, session_id, issue):
    strategies = await agent.execute(
        session_id,
        issue=issue,
        failure_context="Previous attempt failed due to thread safety issues",
    )
    assert strategies is not None


@pytest.mark.asyncio
async def test_strategies_generated_event_payload(agent, repo, session_id, issue):
    strategies = await agent.execute(session_id, issue=issue, num_strategies=3)
    events = repo.get_events(session_id)
    gen_events = [e for e in events if e.event_type == "strategies_generated"]
    assert len(gen_events) == 1
    payload = gen_events[0].payload
    assert payload["count"] == 3
    assert len(payload["strategy_ids"]) == 3
