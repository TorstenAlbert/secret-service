"""Tests for MasterAgent."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ss.agents.master import MasterAgent
from ss.blackboard.database import Database
from ss.blackboard.models import (
    AgentName,
    Mission,
    MemoryType,
    MemoryScope,
    Session,
    Strategy,
)
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.pipeline.events import EventType


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
        "expertise_level": "advanced",
        "domains": ["python", "backend"],
        "communication_style": "technical",
    })
    return m


@pytest.fixture
def mock_memory_mgr():
    m = MagicMock()
    m.recall = MagicMock(return_value=[])
    m.store = MagicMock()
    return m


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def agent(repo, mock_sampling, mock_memory_mgr, mock_vector_store):
    return MasterAgent(repo, mock_sampling, mock_memory_mgr, mock_vector_store)


@pytest.fixture
def session_id(repo):
    s = Session(client_id="client-1", problem_text="test problem")
    repo.insert_session(s)
    return s.id


@pytest.fixture
def strategy(repo, session_id):
    s = Strategy(
        session_id=session_id,
        description="Refactor the memory management module",
        objective="Fix memory leaks",
        rank=1,
    )
    repo.insert_strategy(s)
    return s


@pytest.fixture
def mission(repo, session_id, strategy):
    from ss.blackboard.models import Taktik, TaktikStep
    t = Taktik(
        strategy_id=strategy.id,
        session_id=session_id,
        steps=[TaktikStep(index=0, instruction="Do it", expected_outcome="Done")],
        required_skills=[],
        estimated_complexity="low",
    )
    repo.insert_taktik(t)
    m = Mission(
        taktik_id=t.id,
        strategy_id=strategy.id,
        session_id=session_id,
        status="succeeded",
    )
    repo.insert_mission(m)
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_master_agent_properties(agent):
    assert agent.name == AgentName.master
    assert agent.temperature == 0.3
    assert "Master" in agent.persona


@pytest.mark.asyncio
async def test_join_session_returns_assessment(agent, session_id):
    result = await agent.join_session(
        session_id, issue_summary="App is slow", client_id="client-1"
    )
    assert result["expertise_level"] == "advanced"
    assert "python" in result["domains"]
    assert result["communication_style"] == "technical"


@pytest.mark.asyncio
async def test_join_session_emits_event(agent, repo, session_id):
    await agent.join_session(
        session_id, issue_summary="App is slow", client_id="client-1"
    )
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "master_joined" in event_types


@pytest.mark.asyncio
async def test_execute_dispatches_join_session(agent, session_id):
    result = await agent.execute(
        session_id, issue_summary="App is slow", client_id="client-1"
    )
    assert "expertise_level" in result


@pytest.mark.asyncio
async def test_synthesize_emits_event(agent, repo, session_id, strategy):
    agent._sampling.complete_structured = AsyncMock(return_value={
        "final_answer": "Fix the connection pool",
        "key_insights": ["connection pooling", "resource limits"],
        "recommendations": ["set max_connections=10"],
    })
    await agent.synthesize(
        session_id,
        winning_strategy=strategy,
        score=0.85,
        mission_results=[],
    )
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "master_synthesized" in event_types


@pytest.mark.asyncio
async def test_distribute_learnings_succeeded_creates_memories(
    agent, mock_memory_mgr, session_id, strategy, mission
):
    await agent.distribute_learnings(
        session_id, mission=mission, strategy=strategy, score=0.85
    )
    # Should have called store at least twice (good_practice + pattern + knowledge)
    assert mock_memory_mgr.store.call_count >= 3


@pytest.mark.asyncio
async def test_distribute_learnings_failed_creates_bad_practice(
    agent, mock_memory_mgr, session_id, strategy, mission
):
    mission.status = "failed"
    await agent.distribute_learnings(
        session_id, mission=mission, strategy=strategy, score=0.1
    )
    stored_types = [
        call.args[0].type for call in mock_memory_mgr.store.call_args_list
    ]
    assert MemoryType.bad_practice in stored_types
    assert MemoryType.anti_pattern in stored_types


@pytest.mark.asyncio
async def test_distribute_learnings_always_creates_knowledge(
    agent, mock_memory_mgr, session_id, strategy, mission
):
    await agent.distribute_learnings(
        session_id, mission=mission, strategy=strategy, score=0.5
    )
    stored_types = [
        call.args[0].type for call in mock_memory_mgr.store.call_args_list
    ]
    assert MemoryType.knowledge in stored_types


@pytest.mark.asyncio
async def test_execute_raises_on_unknown_kwargs(agent, session_id):
    with pytest.raises(ValueError):
        await agent.execute(session_id, unknown_kwarg="value")
