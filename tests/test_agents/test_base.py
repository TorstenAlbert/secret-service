"""Tests for BaseAgent abstract class."""
from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from ss.agents.base import BaseAgent
from ss.blackboard.models import AgentName, NoteType
from ss.blackboard.database import Database
from ss.blackboard.repository import Repository
from ss.blackboard.models import Session
from ss.pipeline.events import EventType


# ---------------------------------------------------------------------------
# Concrete test subclass
# ---------------------------------------------------------------------------

class ConcreteAgent(BaseAgent):
    """A minimal concrete agent for testing purposes."""

    @property
    def name(self) -> AgentName:
        return AgentName.reception

    @property
    def persona(self) -> str:
        return "You are a reception agent."

    @property
    def temperature(self) -> float:
        return 0.5

    async def execute(self, session_id: str, **kwargs):
        return {"done": True}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path):
    from ss.config import Config
    config = Config(db_path=tmp_path / "test.db")
    database = Database()
    database.connect(config)
    return database


@pytest.fixture
def repo(db):
    return Repository(db)


@pytest.fixture
def mock_sampling():
    m = MagicMock()
    m.complete = AsyncMock(return_value="LLM response text")
    m.complete_structured = AsyncMock(return_value={"result": "ok"})
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
def agent(repo, mock_sampling, mock_memory_mgr, mock_vector_store):
    return ConcreteAgent(repo, mock_sampling, mock_memory_mgr, mock_vector_store)


@pytest.fixture
def session_id(repo):
    """Insert a dummy session and return its id."""
    s = Session(client_id="client-1", problem_text="test problem")
    repo.insert_session(s)
    return s.id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cannot_instantiate_base_agent_directly():
    """BaseAgent is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseAgent(None, None, None, None)  # type: ignore


def test_concrete_agent_properties(agent):
    assert agent.name == AgentName.reception
    assert agent.persona == "You are a reception agent."
    assert agent.temperature == 0.5


@pytest.mark.asyncio
async def test_execute_returns_result(agent, session_id):
    result = await agent.execute(session_id)
    assert result == {"done": True}


@pytest.mark.asyncio
async def test_llm_call_prepends_persona(agent, mock_sampling):
    await agent.llm_call("Do something.", "Hello")

    mock_sampling.complete.assert_awaited_once()
    call_kwargs = mock_sampling.complete.call_args
    system_prompt = call_kwargs[1]["system_prompt"] if call_kwargs[1] else call_kwargs[0][0]
    assert system_prompt.startswith("You are a reception agent.")
    assert "Do something." in system_prompt


@pytest.mark.asyncio
async def test_llm_call_returns_text(agent):
    result = await agent.llm_call("prompt", "message")
    assert result == "LLM response text"


@pytest.mark.asyncio
async def test_llm_call_structured_prepends_persona(agent, mock_sampling):
    schema = {"type": "object"}
    await agent.llm_call_structured("Structured prompt.", "user msg", schema)

    mock_sampling.complete_structured.assert_awaited_once()
    call_kwargs = mock_sampling.complete_structured.call_args
    system_prompt = call_kwargs[1]["system_prompt"] if call_kwargs[1] else call_kwargs[0][0]
    assert "You are a reception agent." in system_prompt


@pytest.mark.asyncio
async def test_llm_call_structured_returns_dict(agent):
    result = await agent.llm_call_structured("prompt", "msg", {})
    assert result == {"result": "ok"}


def test_emit_event_writes_to_db(agent, repo, session_id):
    agent.emit_event(session_id, EventType.RECEPTION_INTAKE, "intake", {"key": "val"})

    events = repo.get_events(session_id)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "reception_intake"
    assert ev.phase == "intake"
    assert ev.payload == {"key": "val"}
    assert ev.agent_name == AgentName.reception


def test_emit_event_without_phase(agent, repo, session_id):
    agent.emit_event(session_id, EventType.SESSION_COMPLETED, None, {})
    events = repo.get_events(session_id)
    assert events[0].phase is None


def test_write_note_persists(agent, repo, session_id):
    note = agent.write_note(session_id, "An observation about the code.")

    assert note.session_id == session_id
    assert note.content == "An observation about the code."
    assert note.note_type == NoteType.observation
    assert note.agent_name == AgentName.reception

    # Verify it's in the DB
    notes = repo.list_agent_notes(session_id)
    assert len(notes) == 1
    assert notes[0].id == note.id
    assert notes[0].content == note.content


def test_write_note_with_custom_type(agent, repo, session_id):
    note = agent.write_note(session_id, "A concern.", note_type=NoteType.concern)
    assert note.note_type == NoteType.concern

    notes = repo.list_agent_notes(session_id)
    assert notes[0].note_type == NoteType.concern


def test_write_note_with_references(agent, repo, session_id):
    refs = [{"type": "memory", "id": "mem-1"}]
    note = agent.write_note(session_id, "Note with refs.", references=refs)
    assert note.note_references == refs

    notes = repo.list_agent_notes(session_id)
    assert notes[0].note_references == refs


def test_write_note_has_unique_id(agent, repo, session_id):
    note1 = agent.write_note(session_id, "Note 1")
    note2 = agent.write_note(session_id, "Note 2")
    assert note1.id != note2.id


def test_recall_memories_delegates_to_memory_mgr(agent, mock_memory_mgr):
    mock_memory_mgr.recall.return_value = ["memory1", "memory2"]
    result = agent.recall_memories("some query", limit=3)

    mock_memory_mgr.recall.assert_called_once_with("some query", type=None, scope=None, limit=3)
    assert result == ["memory1", "memory2"]


def test_recall_memories_with_filters(agent, mock_memory_mgr):
    from ss.blackboard.models import MemoryType, MemoryScope
    agent.recall_memories("query", type=MemoryType.knowledge, scope=MemoryScope.long_term)

    mock_memory_mgr.recall.assert_called_once_with(
        "query",
        type=MemoryType.knowledge,
        scope=MemoryScope.long_term,
        limit=5,
    )
