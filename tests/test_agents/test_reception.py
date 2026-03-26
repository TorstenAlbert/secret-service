"""Tests for ReceptionAgent."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ss.agents.reception import ReceptionAgent
from ss.blackboard.database import Database
from ss.blackboard.models import IssueClassification, Session
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
        "summary": "Memory leak in user service",
        "classification": "bug",
        "severity": "high",
        "who": "backend engineers",
        "where_location": "user-service/memory.py",
        "why_reason": "Unclosed database connections",
        "precondition": "Service is running under load",
        "postcondition": "Service crashes with OOM error",
        "key_points": ["OOM", "memory leak", "database connections"],
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
    return ReceptionAgent(repo, mock_sampling, mock_memory_mgr, mock_vector_store)


@pytest.fixture
def session_id(repo):
    s = Session(client_id="client-1", problem_text="test problem")
    repo.insert_session(s)
    return s.id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reception_agent_properties(agent):
    from ss.blackboard.models import AgentName
    assert agent.name == AgentName.reception
    assert agent.temperature == 0.1
    assert "Reception" in agent.persona


@pytest.mark.asyncio
async def test_execute_returns_issue(agent, session_id):
    issue = await agent.execute(session_id, problem="App crashes on startup")
    assert issue is not None
    assert issue.session_id == session_id
    assert issue.summary == "Memory leak in user service"
    assert issue.classification == IssueClassification.bug
    assert issue.severity == "high"
    assert issue.who == "backend engineers"
    assert len(issue.key_points) == 3


@pytest.mark.asyncio
async def test_execute_persists_issue(agent, repo, session_id):
    issue = await agent.execute(session_id, problem="App crashes on startup")
    issues = repo.get_issue_by_session(session_id)
    assert len(issues) == 1
    assert issues[0].id == issue.id


@pytest.mark.asyncio
async def test_execute_indexes_vector(agent, mock_vector_store, session_id):
    issue = await agent.execute(session_id, problem="App crashes on startup")
    mock_vector_store.index.assert_called_once_with("issue", issue.id, issue.summary)


@pytest.mark.asyncio
async def test_execute_emits_events(agent, repo, session_id):
    await agent.execute(session_id, problem="App crashes on startup")
    events = repo.get_events(session_id)
    event_types = [e.event_type for e in events]
    assert "agent_started" in event_types
    assert "reception_intake" in event_types


@pytest.mark.asyncio
async def test_execute_recalls_memories(agent, mock_memory_mgr, session_id):
    await agent.execute(session_id, problem="memory leak")
    mock_memory_mgr.recall.assert_called()


@pytest.mark.asyncio
async def test_execute_with_context(agent, session_id):
    """context kwarg should be included in the prompt without errors."""
    issue = await agent.execute(
        session_id,
        problem="Something broken",
        context={"env": "production", "version": "1.2.3"},
    )
    assert issue is not None


@pytest.mark.asyncio
async def test_reception_intake_event_payload(agent, repo, session_id):
    issue = await agent.execute(session_id, problem="App crashes on startup")
    events = repo.get_events(session_id)
    intake_events = [e for e in events if e.event_type == "reception_intake"]
    assert len(intake_events) == 1
    payload = intake_events[0].payload
    assert payload["issue_id"] == issue.id
    assert payload["classification"] == "bug"
    assert payload["severity"] == "high"
