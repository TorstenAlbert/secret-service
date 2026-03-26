"""Tests for MissionAgent."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ss.agents.mission import MissionAgent
from ss.blackboard.database import Database
from ss.blackboard.models import (
    AgentName,
    Session,
    Strategy,
    Taktik,
    TaktikStep,
)
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
def mock_sampling_success():
    m = MagicMock()
    m.complete = AsyncMock(return_value="text")
    # Returns success for each step
    m.complete_structured = AsyncMock(return_value={
        "action": "Performed the action successfully",
        "actual_outcome": "All connections are now managed properly",
        "success": True,
        "error_detail": None,
    })
    return m


@pytest.fixture
def mock_sampling_failure():
    m = MagicMock()
    m.complete = AsyncMock(return_value="text")
    # Returns failure on the first call, success on subsequent (to test stop-on-failure)
    side_effects = [
        {
            "action": "Attempted to wrap connections",
            "actual_outcome": "Import error: module not found",
            "success": False,
            "error_detail": "ModuleNotFoundError: No module named 'poolmanager'",
        },
        {
            "action": "This should not be called",
            "actual_outcome": "Should not reach here",
            "success": True,
        },
    ]
    m.complete_structured = AsyncMock(side_effect=side_effects)
    return m


@pytest.fixture
def mock_memory_mgr():
    return MagicMock()


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def agent_success(repo, mock_sampling_success, mock_memory_mgr, mock_vector_store):
    return MissionAgent(repo, mock_sampling_success, mock_memory_mgr, mock_vector_store)


@pytest.fixture
def agent_failure(repo, mock_sampling_failure, mock_memory_mgr, mock_vector_store):
    return MissionAgent(repo, mock_sampling_failure, mock_memory_mgr, mock_vector_store)


@pytest.fixture
def session_id(repo):
    s = Session(client_id="client-1", problem_text="test problem")
    repo.insert_session(s)
    return s.id


@pytest.fixture
def strategy(repo, session_id):
    s = Strategy(
        session_id=session_id,
        description="Fix memory leaks with connection pooling",
        objective="Eliminate OOM errors",
        rank=1,
    )
    repo.insert_strategy(s)
    return s


@pytest.fixture
def taktik(repo, session_id, strategy):
    t = Taktik(
        strategy_id=strategy.id,
        session_id=session_id,
        steps=[
            TaktikStep(index=0, instruction="Audit connections", expected_outcome="List of issues"),
            TaktikStep(index=1, instruction="Wrap with context managers", expected_outcome="Clean code"),
            TaktikStep(index=2, instruction="Run tests", expected_outcome="All tests pass"),
        ],
        required_skills=["Python"],
        estimated_complexity="medium",
    )
    repo.insert_taktik(t)
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mission_agent_properties(agent_success):
    assert agent_success.name == AgentName.mission
    assert agent_success.temperature == 0.2
    assert "Mission" in agent_success.persona


@pytest.mark.asyncio
async def test_execute_returns_mission(agent_success, session_id, taktik, strategy):
    mission = await agent_success.execute(
        session_id, taktik=taktik, strategy_id=strategy.id
    )
    assert mission is not None
    assert mission.taktik_id == taktik.id
    assert mission.strategy_id == strategy.id
    assert mission.session_id == session_id


@pytest.mark.asyncio
async def test_execute_success_mission_status(agent_success, session_id, taktik, strategy):
    mission = await agent_success.execute(
        session_id, taktik=taktik, strategy_id=strategy.id
    )
    assert mission.status == "succeeded"


@pytest.mark.asyncio
async def test_execute_failure_mission_status(agent_failure, session_id, taktik, strategy):
    mission = await agent_failure.execute(
        session_id, taktik=taktik, strategy_id=strategy.id
    )
    assert mission.status == "failed"


@pytest.mark.asyncio
async def test_execute_failure_stops_on_first_failure(agent_failure, repo, session_id, taktik, strategy):
    """Only one step result should be created (stopped at failure)."""
    mission = await agent_failure.execute(
        session_id, taktik=taktik, strategy_id=strategy.id
    )
    results = repo.list_mission_results(mission.id)
    assert len(results) == 1
    assert results[0].success is False


@pytest.mark.asyncio
async def test_execute_success_creates_all_results(agent_success, repo, session_id, taktik, strategy):
    mission = await agent_success.execute(
        session_id, taktik=taktik, strategy_id=strategy.id
    )
    results = repo.list_mission_results(mission.id)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_execute_persists_mission_in_db(agent_success, repo, session_id, taktik, strategy):
    mission = await agent_success.execute(
        session_id, taktik=taktik, strategy_id=strategy.id
    )
    missions = repo.list_missions_by_session(session_id)
    assert len(missions) == 1
    assert missions[0].id == mission.id
    assert missions[0].status == "succeeded"


@pytest.mark.asyncio
async def test_execute_emits_mission_events(agent_success, repo, session_id, taktik, strategy):
    await agent_success.execute(
        session_id, taktik=taktik, strategy_id=strategy.id
    )
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "mission_started" in event_types
    assert "mission_step" in event_types
    assert "mission_completed" in event_types


@pytest.mark.asyncio
async def test_execute_emits_step_events_per_step(agent_success, repo, session_id, taktik, strategy):
    await agent_success.execute(
        session_id, taktik=taktik, strategy_id=strategy.id
    )
    events = repo.get_events(session_id)
    step_events = [e for e in events if e.event_type == "mission_step"]
    assert len(step_events) == 3  # One per step


@pytest.mark.asyncio
async def test_execute_mission_has_duration(agent_success, session_id, taktik, strategy):
    mission = await agent_success.execute(
        session_id, taktik=taktik, strategy_id=strategy.id
    )
    assert mission.completed_at is not None
    assert mission.duration_ms is not None
    assert mission.duration_ms >= 0
