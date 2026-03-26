"""Tests for JudgeAgent."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ss.agents.judge import JudgeAgent
from ss.blackboard.database import Database
from ss.blackboard.models import (
    AgentName,
    Issue,
    IssueClassification,
    NoteType,
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
def mock_sampling_verified():
    m = MagicMock()
    m.complete = AsyncMock(return_value="text")
    m.complete_structured = AsyncMock(return_value={
        "verified": True,
        "reasoning": "The plan looks solid and addresses all identified issues.",
        "issues_found": [],
        "suggestions": ["Consider adding logging"],
    })
    return m


@pytest.fixture
def mock_sampling_rejected():
    m = MagicMock()
    m.complete = AsyncMock(return_value="text")
    m.complete_structured = AsyncMock(return_value={
        "verified": False,
        "reasoning": "Step 2 is missing error handling.",
        "issues_found": ["No error handling in step 2", "Missing rollback logic"],
        "suggestions": ["Add try/except in step 2", "Implement transaction rollback"],
    })
    return m


@pytest.fixture
def mock_memory_mgr():
    m = MagicMock()
    m.recall = MagicMock(return_value=[])
    return m


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def agent_verified(repo, mock_sampling_verified, mock_memory_mgr, mock_vector_store):
    return JudgeAgent(repo, mock_sampling_verified, mock_memory_mgr, mock_vector_store)


@pytest.fixture
def agent_rejected(repo, mock_sampling_rejected, mock_memory_mgr, mock_vector_store):
    return JudgeAgent(repo, mock_sampling_rejected, mock_memory_mgr, mock_vector_store)


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
        who="engineers",
        where_location="user-service",
        why_reason="Unclosed connections",
        precondition="Under load",
        postcondition="OOM",
        key_points=["OOM"],
    )
    repo.insert_issue(i)
    return i


@pytest.fixture
def strategy(repo, session_id):
    s = Strategy(
        session_id=session_id,
        description="Add connection pooling",
        objective="Fix memory leaks",
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
            TaktikStep(index=1, instruction="Fix with context managers", expected_outcome="Clean code"),
        ],
        required_skills=["Python"],
        estimated_complexity="medium",
    )
    repo.insert_taktik(t)
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_judge_agent_properties(agent_verified):
    assert agent_verified.name == AgentName.judge
    assert agent_verified.temperature == 0.1
    assert "Judge" in agent_verified.persona


@pytest.mark.asyncio
async def test_execute_returns_verification_dict(agent_verified, session_id, taktik, strategy, issue):
    result = await agent_verified.execute(
        session_id, taktik=taktik, strategy=strategy, issue=issue
    )
    assert isinstance(result, dict)
    assert "verified" in result
    assert result["verified"] is True


@pytest.mark.asyncio
async def test_execute_updates_taktik_verification(agent_verified, repo, session_id, taktik, strategy, issue):
    await agent_verified.execute(
        session_id, taktik=taktik, strategy=strategy, issue=issue
    )
    # Verify the taktik was updated in DB
    row = repo._conn.execute(
        "SELECT verified, judge_verification FROM taktiks WHERE id = ?", (taktik.id,)
    ).fetchone()
    assert bool(row["verified"]) is True
    assert row["judge_verification"] is not None


@pytest.mark.asyncio
async def test_execute_emits_verified_event(agent_verified, repo, session_id, taktik, strategy, issue):
    await agent_verified.execute(
        session_id, taktik=taktik, strategy=strategy, issue=issue
    )
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "judge_verified" in event_types


@pytest.mark.asyncio
async def test_execute_rejected_writes_note(agent_rejected, repo, session_id, taktik, strategy, issue):
    await agent_rejected.execute(
        session_id, taktik=taktik, strategy=strategy, issue=issue
    )
    notes = repo.list_agent_notes(session_id, agent_name=AgentName.judge)
    assert len(notes) == 1
    assert notes[0].note_type == NoteType.error_analysis


@pytest.mark.asyncio
async def test_execute_verified_does_not_write_note(agent_verified, repo, session_id, taktik, strategy, issue):
    await agent_verified.execute(
        session_id, taktik=taktik, strategy=strategy, issue=issue
    )
    notes = repo.list_agent_notes(session_id, agent_name=AgentName.judge)
    assert len(notes) == 0


@pytest.mark.asyncio
async def test_execute_rejected_payload_has_rejection_reason(
    agent_rejected, repo, session_id, taktik, strategy, issue
):
    await agent_rejected.execute(
        session_id, taktik=taktik, strategy=strategy, issue=issue
    )
    events = repo.get_events(session_id)
    verified_events = [e for e in events if e.event_type == "judge_verified"]
    assert len(verified_events) == 1
    payload = verified_events[0].payload
    assert "rejection_reason" in payload
    assert payload["verified"] is False


@pytest.mark.asyncio
async def test_execute_recalls_bad_practice_memories(agent_verified, mock_memory_mgr, session_id, taktik, strategy, issue):
    from ss.blackboard.models import MemoryType
    await agent_verified.execute(
        session_id, taktik=taktik, strategy=strategy, issue=issue
    )
    mock_memory_mgr.recall.assert_called()
    # Should recall bad_practice
    calls = mock_memory_mgr.recall.call_args_list
    bad_practice_calls = [c for c in calls if c[1].get("type") == MemoryType.bad_practice]
    assert len(bad_practice_calls) >= 1
