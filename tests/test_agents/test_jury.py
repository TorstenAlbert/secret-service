"""Tests for JuryAgent."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ss.agents.jury import JuryAgent
from ss.blackboard.database import Database
from ss.blackboard.models import (
    AgentName,
    Mission,
    MissionResult,
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
def config(tmp_path: Path):
    return Config(db_path=tmp_path / "test.db")


@pytest.fixture
def db(config):
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
        "correctness": 0.9,
        "completeness": 0.85,
        "elegance": 0.8,
        "robustness": 0.75,
        "efficiency": 0.7,
        "reasoning": "The solution is correct and well-structured.",
    })
    return m


@pytest.fixture
def mock_memory_mgr():
    return MagicMock()


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def agent(repo, mock_sampling, mock_memory_mgr, mock_vector_store, config):
    return JuryAgent(repo, mock_sampling, mock_memory_mgr, mock_vector_store, config=config)


@pytest.fixture
def session_id(repo):
    s = Session(client_id="client-1", problem_text="test problem")
    repo.insert_session(s)
    return s.id


@pytest.fixture
def strategy(repo, session_id):
    s = Strategy(
        session_id=session_id,
        description="Fix memory leaks via connection pooling",
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
            TaktikStep(index=0, instruction="Audit", expected_outcome="List of issues"),
        ],
        required_skills=["Python"],
        estimated_complexity="low",
    )
    repo.insert_taktik(t)
    return t


@pytest.fixture
def succeeded_mission(repo, session_id, strategy, taktik):
    m = Mission(
        taktik_id=taktik.id,
        strategy_id=strategy.id,
        session_id=session_id,
        status="succeeded",
    )
    repo.insert_mission(m)
    # Add a result
    result = MissionResult(
        mission_id=m.id,
        step_index=0,
        action="Audited the codebase",
        expected_outcome="List of issues",
        actual_outcome="Found 5 unclosed connections",
        success=True,
    )
    repo.insert_mission_result(result)
    return m


@pytest.fixture
def failed_mission(repo, session_id, strategy, taktik):
    m = Mission(
        taktik_id=taktik.id,
        strategy_id=strategy.id,
        session_id=session_id,
        status="failed",
    )
    repo.insert_mission(m)
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_jury_agent_properties(agent):
    assert agent.name == AgentName.jury
    assert agent.temperature == 0.2
    assert "Jury" in agent.persona


@pytest.mark.asyncio
async def test_execute_returns_scores_for_succeeded(agent, session_id, succeeded_mission):
    scores = await agent.execute(session_id, missions=[succeeded_mission])
    assert len(scores) == 1


@pytest.mark.asyncio
async def test_execute_skips_failed_missions(agent, session_id, failed_mission):
    scores = await agent.execute(session_id, missions=[failed_mission])
    assert len(scores) == 0


@pytest.mark.asyncio
async def test_execute_score_fields(agent, session_id, succeeded_mission):
    scores = await agent.execute(session_id, missions=[succeeded_mission])
    score = scores[0]
    assert score.correctness == 0.9
    assert score.completeness == 0.85
    assert score.elegance == 0.8
    assert score.robustness == 0.75
    assert score.efficiency == 0.7
    assert score.reasoning == "The solution is correct and well-structured."


@pytest.mark.asyncio
async def test_execute_calculates_weighted_total(agent, session_id, succeeded_mission, config):
    scores = await agent.execute(session_id, missions=[succeeded_mission])
    score = scores[0]
    # Expected: 0.9*0.30 + 0.85*0.25 + 0.75*0.20 + 0.8*0.15 + 0.7*0.10
    weights = config.score_weights
    expected = (
        0.9 * weights["correctness"]
        + 0.85 * weights["completeness"]
        + 0.75 * weights["robustness"]
        + 0.8 * weights["elegance"]
        + 0.7 * weights["efficiency"]
    )
    assert abs(score.weighted_total - expected) < 1e-6


@pytest.mark.asyncio
async def test_execute_persists_strategy_score(agent, repo, session_id, succeeded_mission):
    scores = await agent.execute(session_id, missions=[succeeded_mission])
    row = repo._conn.execute(
        "SELECT * FROM strategy_scores WHERE id = ?", (scores[0].id,)
    ).fetchone()
    assert row is not None
    assert row["session_id"] == session_id


@pytest.mark.asyncio
async def test_execute_updates_strategy_rating_proven(agent, repo, session_id, succeeded_mission, strategy):
    """Score of ~0.82 should be rated 'proven' (>= 0.75 threshold)."""
    scores = await agent.execute(session_id, missions=[succeeded_mission])
    updated = repo.get_strategy(strategy.id)
    assert updated.rating_label == "proven"
    assert updated.jury_score is not None


@pytest.mark.asyncio
async def test_execute_updates_strategy_rating_adequate(
    repo, mock_memory_mgr, mock_vector_store, session_id, succeeded_mission, strategy, config
):
    """Score in [0.3, 0.75) should be 'adequate'."""
    mock = MagicMock()
    mock.complete = AsyncMock(return_value="text")
    mock.complete_structured = AsyncMock(return_value={
        "correctness": 0.5,
        "completeness": 0.5,
        "elegance": 0.5,
        "robustness": 0.5,
        "efficiency": 0.5,
        "reasoning": "Adequate solution.",
    })
    agent = JuryAgent(repo, mock, mock_memory_mgr, mock_vector_store, config=config)
    scores = await agent.execute(session_id, missions=[succeeded_mission])
    updated = repo.get_strategy(strategy.id)
    assert updated.rating_label == "adequate"


@pytest.mark.asyncio
async def test_execute_emits_jury_scored_event(agent, repo, session_id, succeeded_mission):
    await agent.execute(session_id, missions=[succeeded_mission])
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "jury_scored" in event_types


@pytest.mark.asyncio
async def test_execute_jury_event_payload(agent, repo, session_id, succeeded_mission, strategy):
    scores = await agent.execute(session_id, missions=[succeeded_mission])
    events = repo.get_events(session_id)
    jury_events = [e for e in events if e.event_type == "jury_scored"]
    assert len(jury_events) == 1
    payload = jury_events[0].payload
    assert payload["strategy_id"] == strategy.id
    assert "weighted_total" in payload
    assert "rating_label" in payload


@pytest.mark.asyncio
async def test_execute_empty_missions_returns_empty(agent, session_id):
    scores = await agent.execute(session_id, missions=[])
    assert scores == []


@pytest.mark.asyncio
async def test_execute_multiple_succeeded_missions(
    repo, mock_memory_mgr, mock_vector_store, session_id, config
):
    """Multiple succeeded missions should all be scored."""
    mock = MagicMock()
    mock.complete = AsyncMock(return_value="text")
    mock.complete_structured = AsyncMock(return_value={
        "correctness": 0.8,
        "completeness": 0.8,
        "elegance": 0.8,
        "robustness": 0.8,
        "efficiency": 0.8,
        "reasoning": "Good.",
    })
    agent = JuryAgent(repo, mock, mock_memory_mgr, mock_vector_store, config=config)

    # Create two strategies + missions
    missions = []
    for i in range(2):
        s = Strategy(
            session_id=session_id,
            description=f"Strategy {i}",
            objective="Fix it",
            rank=i + 1,
        )
        repo.insert_strategy(s)
        t = Taktik(
            strategy_id=s.id,
            session_id=session_id,
            steps=[TaktikStep(index=0, instruction="Do it", expected_outcome="Done")],
            required_skills=[],
            estimated_complexity="low",
        )
        repo.insert_taktik(t)
        m = Mission(
            taktik_id=t.id,
            strategy_id=s.id,
            session_id=session_id,
            status="succeeded",
        )
        repo.insert_mission(m)
        result = MissionResult(
            mission_id=m.id,
            step_index=0,
            action="Did it",
            actual_outcome="Done",
            success=True,
        )
        repo.insert_mission_result(result)
        missions.append(m)

    scores = await agent.execute(session_id, missions=missions)
    assert len(scores) == 2
