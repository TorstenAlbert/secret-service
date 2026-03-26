"""Tests for TaktikPlannerAgent."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ss.agents.taktik_planner import TaktikPlannerAgent
from ss.blackboard.database import Database
from ss.blackboard.models import AgentName, Session, Strategy
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
        "steps": [
            {
                "index": 0,
                "instruction": "Identify all unclosed connections in the codebase",
                "expected_outcome": "List of files and line numbers with issues",
            },
            {
                "index": 1,
                "instruction": "Wrap database calls in context managers",
                "expected_outcome": "All DB calls use 'with' statements",
            },
            {
                "index": 2,
                "instruction": "Add connection pool configuration",
                "expected_outcome": "Pool size limits enforced",
            },
        ],
        "required_skills": ["Python", "SQLAlchemy", "async programming"],
        "estimated_complexity": "medium",
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
    return TaktikPlannerAgent(repo, mock_sampling, mock_memory_mgr, mock_vector_store)


@pytest.fixture
def session_id(repo):
    s = Session(client_id="client-1", problem_text="test problem")
    repo.insert_session(s)
    return s.id


@pytest.fixture
def strategy(repo, session_id):
    s = Strategy(
        session_id=session_id,
        description="Add connection pooling to the user service",
        objective="Fix memory leaks",
        approach_type="refactor",
        rank=1,
        confidence=0.9,
    )
    repo.insert_strategy(s)
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_taktik_planner_properties(agent):
    assert agent.name == AgentName.taktik_planner
    assert agent.temperature == 0.8
    assert "Taktik Planner" in agent.persona or "taktik" in agent.persona.lower()


@pytest.mark.asyncio
async def test_execute_returns_taktik(agent, session_id, strategy):
    taktik = await agent.execute(session_id, strategy=strategy)
    assert taktik is not None
    assert taktik.strategy_id == strategy.id
    assert taktik.session_id == session_id


@pytest.mark.asyncio
async def test_execute_taktik_has_steps(agent, session_id, strategy):
    taktik = await agent.execute(session_id, strategy=strategy)
    assert len(taktik.steps) == 3
    assert taktik.steps[0].index == 0
    assert taktik.steps[1].index == 1


@pytest.mark.asyncio
async def test_execute_taktik_has_skills(agent, session_id, strategy):
    taktik = await agent.execute(session_id, strategy=strategy)
    assert "Python" in taktik.required_skills
    assert taktik.estimated_complexity == "medium"


@pytest.mark.asyncio
async def test_execute_persists_taktik(agent, repo, session_id, strategy):
    taktik = await agent.execute(session_id, strategy=strategy)
    # Verify it can be retrieved via related strategy
    from ss.blackboard.models import Taktik
    # fetch directly from db
    row = repo._conn.execute(
        "SELECT * FROM taktiks WHERE id = ?", (taktik.id,)
    ).fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_execute_indexes_vector(agent, mock_vector_store, session_id, strategy):
    taktik = await agent.execute(session_id, strategy=strategy)
    mock_vector_store.index.assert_called_once()
    call_args = mock_vector_store.index.call_args
    assert call_args[0][0] == "taktik"
    assert call_args[0][1] == taktik.id


@pytest.mark.asyncio
async def test_execute_emits_taktik_planned_event(agent, repo, session_id, strategy):
    await agent.execute(session_id, strategy=strategy)
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "taktik_planned" in event_types


@pytest.mark.asyncio
async def test_execute_with_attempt_number(agent, session_id, strategy):
    taktik = await agent.execute(session_id, strategy=strategy, attempt=2)
    assert taktik.attempt_number == 2


@pytest.mark.asyncio
async def test_execute_with_rejection_reason(agent, session_id, strategy):
    """Rejection reason should not cause errors."""
    taktik = await agent.execute(
        session_id,
        strategy=strategy,
        attempt=2,
        rejection_reason="Missing error handling in step 2",
    )
    assert taktik is not None


@pytest.mark.asyncio
async def test_execute_recalls_knowledge_memories(agent, mock_memory_mgr, session_id, strategy):
    from ss.blackboard.models import MemoryType
    await agent.execute(session_id, strategy=strategy)
    mock_memory_mgr.recall.assert_called()
    call_kwargs = mock_memory_mgr.recall.call_args
    assert call_kwargs[1].get("type") == MemoryType.knowledge


@pytest.mark.asyncio
async def test_taktik_planned_event_payload(agent, repo, session_id, strategy):
    taktik = await agent.execute(session_id, strategy=strategy)
    events = repo.get_events(session_id)
    planned_events = [e for e in events if e.event_type == "taktik_planned"]
    assert len(planned_events) == 1
    payload = planned_events[0].payload
    assert payload["taktik_id"] == taktik.id
    assert payload["strategy_id"] == strategy.id
    assert payload["step_count"] == 3
