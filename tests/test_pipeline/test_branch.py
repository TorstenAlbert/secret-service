"""Tests for StrategyBranch."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ss.blackboard.database import Database
from ss.blackboard.models import (
    AgentName,
    Issue,
    IssueClassification,
    Mission,
    Session,
    Strategy,
    Taktik,
    TaktikStep,
)
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.pipeline.branch import StrategyBranch, StrategyExhausted


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
def config():
    return Config(max_judge_retries=3)


@pytest.fixture
def session_id(repo):
    s = Session(client_id="test-client", problem_text="Test problem")
    repo.insert_session(s)
    return s.id


@pytest.fixture
def issue(repo, session_id):
    i = Issue(
        session_id=session_id,
        summary="Test issue summary",
        classification=IssueClassification.bug,
        severity="high",
        who="Developer",
        where_location="backend",
        why_reason="null pointer",
        precondition="Service running",
        postcondition="Bug fixed",
        key_points=["point 1", "point 2"],
    )
    repo.insert_issue(i)
    return i


@pytest.fixture
def strategy(repo, session_id):
    s = Strategy(
        session_id=session_id,
        description="Fix the null pointer by adding null checks",
        objective="Prevent NPE crashes",
        approach_type="bug_fix",
        rank=1,
        confidence=0.9,
    )
    repo.insert_strategy(s)
    return s


def _make_taktik_response():
    return {
        "steps": [
            {
                "index": 0,
                "instruction": "Add null check before dereferencing",
                "expected_outcome": "No NPE thrown",
            }
        ],
        "required_skills": ["Python"],
        "estimated_complexity": "low",
    }


def _make_verified_judge_response():
    return {
        "verified": True,
        "reasoning": "Plan is solid",
        "issues_found": [],
        "suggestions": [],
    }


def _make_rejected_judge_response(reason="Incomplete plan"):
    return {
        "verified": False,
        "reasoning": reason,
        "issues_found": [reason],
        "suggestions": ["Add more steps"],
    }


def _make_mission_step_response():
    return {
        "action": "Added null check",
        "actual_outcome": "Code updated",
        "success": True,
    }


@pytest.fixture
def mock_sampling_success():
    """Sampling mock that always verifies and succeeds."""
    m = MagicMock()
    m.complete = AsyncMock(return_value="text")
    m.complete_structured = AsyncMock(
        side_effect=[
            _make_taktik_response(),       # taktik_planner
            _make_verified_judge_response(),  # judge
            _make_mission_step_response(),    # mission step 0
        ]
    )
    return m


@pytest.fixture
def mock_sampling_retry_then_success():
    """Sampling mock that rejects once then verifies."""
    m = MagicMock()
    m.complete = AsyncMock(return_value="text")
    m.complete_structured = AsyncMock(
        side_effect=[
            _make_taktik_response(),          # attempt 1: taktik_planner
            _make_rejected_judge_response(),  # attempt 1: judge — rejected
            _make_taktik_response(),          # attempt 2: taktik_planner
            _make_verified_judge_response(),  # attempt 2: judge — verified
            _make_mission_step_response(),    # mission step 0
        ]
    )
    return m


@pytest.fixture
def mock_sampling_always_rejected():
    """Sampling mock that always rejects."""
    m = MagicMock()
    m.complete = AsyncMock(return_value="text")
    # taktik + judge repeated for max_judge_retries times
    side_effects = []
    for _ in range(3):  # max_judge_retries=3
        side_effects.append(_make_taktik_response())
        side_effects.append(_make_rejected_judge_response("Always fails"))
    m.complete_structured = AsyncMock(side_effect=side_effects)
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
    m.search = MagicMock(return_value=[])
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_branch_returns_mission_on_success(
    repo, session_id, issue, strategy, mock_sampling_success, mock_memory_mgr,
    mock_vector_store, config
):
    """A verified taktik should lead to a returned Mission."""
    branch = StrategyBranch(repo, mock_sampling_success, mock_memory_mgr, mock_vector_store, config)
    mission = await branch.run(session_id, strategy)
    assert isinstance(mission, Mission)
    assert mission.strategy_id == strategy.id
    assert mission.session_id == session_id


@pytest.mark.asyncio
async def test_branch_mission_status_succeeded(
    repo, session_id, issue, strategy, mock_sampling_success, mock_memory_mgr,
    mock_vector_store, config
):
    """Mission status should be 'succeeded' when all steps pass."""
    branch = StrategyBranch(repo, mock_sampling_success, mock_memory_mgr, mock_vector_store, config)
    mission = await branch.run(session_id, strategy)
    assert mission.status == "succeeded"


@pytest.mark.asyncio
async def test_branch_emits_taktik_planned_event(
    repo, session_id, issue, strategy, mock_sampling_success, mock_memory_mgr,
    mock_vector_store, config
):
    """TAKTIK_PLANNED event should be emitted during the branch run."""
    branch = StrategyBranch(repo, mock_sampling_success, mock_memory_mgr, mock_vector_store, config)
    await branch.run(session_id, strategy)
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "taktik_planned" in event_types


@pytest.mark.asyncio
async def test_branch_emits_judge_verified_event(
    repo, session_id, issue, strategy, mock_sampling_success, mock_memory_mgr,
    mock_vector_store, config
):
    """JUDGE_VERIFIED event should be emitted."""
    branch = StrategyBranch(repo, mock_sampling_success, mock_memory_mgr, mock_vector_store, config)
    await branch.run(session_id, strategy)
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "judge_verified" in event_types


@pytest.mark.asyncio
async def test_branch_emits_mission_completed_event(
    repo, session_id, issue, strategy, mock_sampling_success, mock_memory_mgr,
    mock_vector_store, config
):
    """MISSION_COMPLETED event should be emitted after execution."""
    branch = StrategyBranch(repo, mock_sampling_success, mock_memory_mgr, mock_vector_store, config)
    await branch.run(session_id, strategy)
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "mission_completed" in event_types


@pytest.mark.asyncio
async def test_branch_retries_on_rejection(
    repo, session_id, issue, strategy, mock_sampling_retry_then_success, mock_memory_mgr,
    mock_vector_store, config
):
    """Branch should retry when judge rejects and succeed on second attempt."""
    branch = StrategyBranch(
        repo, mock_sampling_retry_then_success, mock_memory_mgr, mock_vector_store, config
    )
    mission = await branch.run(session_id, strategy)
    assert isinstance(mission, Mission)
    assert mission.status == "succeeded"


@pytest.mark.asyncio
async def test_branch_emits_rejection_loop_event(
    repo, session_id, issue, strategy, mock_sampling_retry_then_success, mock_memory_mgr,
    mock_vector_store, config
):
    """JUDGE_REJECTED_LOOP event should be emitted on first rejection."""
    branch = StrategyBranch(
        repo, mock_sampling_retry_then_success, mock_memory_mgr, mock_vector_store, config
    )
    await branch.run(session_id, strategy)
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "judge_rejected_loop" in event_types


@pytest.mark.asyncio
async def test_branch_raises_strategy_exhausted_after_all_retries(
    repo, session_id, issue, strategy, mock_sampling_always_rejected, mock_memory_mgr,
    mock_vector_store, config
):
    """StrategyExhausted should be raised when all retries fail."""
    branch = StrategyBranch(
        repo, mock_sampling_always_rejected, mock_memory_mgr, mock_vector_store, config
    )
    with pytest.raises(StrategyExhausted) as exc_info:
        await branch.run(session_id, strategy)
    assert exc_info.value.strategy_id == strategy.id
    assert exc_info.value.reason is not None


@pytest.mark.asyncio
async def test_strategy_exhausted_has_attributes():
    """StrategyExhausted exception stores strategy_id and reason."""
    exc = StrategyExhausted("strat-123", "not feasible")
    assert exc.strategy_id == "strat-123"
    assert exc.reason == "not feasible"
    assert "strat-123" in str(exc)


@pytest.mark.asyncio
async def test_branch_multiple_rejection_events(
    repo, session_id, issue, strategy, mock_sampling_always_rejected, mock_memory_mgr,
    mock_vector_store, config
):
    """Three rejection loop events should be emitted for 3 retries."""
    branch = StrategyBranch(
        repo, mock_sampling_always_rejected, mock_memory_mgr, mock_vector_store, config
    )
    with pytest.raises(StrategyExhausted):
        await branch.run(session_id, strategy)

    events = repo.get_events(session_id)
    rejection_events = [e for e in events if e.event_type == "judge_rejected_loop"]
    assert len(rejection_events) == config.max_judge_retries


@pytest.mark.asyncio
async def test_branch_mission_is_persisted(
    repo, session_id, issue, strategy, mock_sampling_success, mock_memory_mgr,
    mock_vector_store, config
):
    """Completed mission should be persisted in the database."""
    branch = StrategyBranch(repo, mock_sampling_success, mock_memory_mgr, mock_vector_store, config)
    mission = await branch.run(session_id, strategy)

    missions = repo.list_missions_by_session(session_id)
    mission_ids = [m.id for m in missions]
    assert mission.id in mission_ids
