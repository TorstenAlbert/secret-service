"""Abstract base class for all SS agents."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from ss.blackboard.models import AgentName, AgentNote, NoteType
from ss.blackboard.repository import Repository
from ss.pipeline.events import EventType, create_event
from ss.sampling.adapter import SamplingAdapter


class BaseAgent(ABC):
    """Abstract base class for all SS pipeline agents.

    Subclasses must define the ``name``, ``persona``, and ``temperature``
    properties and implement the ``execute`` method.
    """

    def __init__(
        self,
        repo: Repository,
        sampling: SamplingAdapter,
        memory_mgr: Any,
        vector_store: Any,
    ) -> None:
        self._repo = repo
        self._sampling = sampling
        self._memory_mgr = memory_mgr
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    # Abstract properties — set by subclasses
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> AgentName:
        """The agent's identifier."""

    @property
    @abstractmethod
    def persona(self) -> str:
        """System persona prepended to every LLM call."""

    @property
    @abstractmethod
    def temperature(self) -> float:
        """Default sampling temperature."""

    # ------------------------------------------------------------------
    # Abstract execution
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(self, session_id: str, **kwargs: Any) -> Any:
        """Run the agent's main logic for a given session."""

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def llm_call(self, system_prompt: str, user_message: str) -> str:
        """Call the LLM with persona prepended to the system prompt.

        Returns the raw text response.
        """
        full_system = f"{self.persona}\n\n{system_prompt}"
        return await self._sampling.complete(
            system_prompt=full_system,
            messages=[{"role": "user", "content": user_message}],
            temperature=self.temperature,
        )

    async def llm_call_structured(
        self, system_prompt: str, user_message: str, schema: dict
    ) -> dict:
        """Call the LLM expecting a structured JSON response."""
        full_system = f"{self.persona}\n\n{system_prompt}"
        return await self._sampling.complete_structured(
            system_prompt=full_system,
            messages=[{"role": "user", "content": user_message}],
            schema=schema,
            temperature=self.temperature,
        )

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def emit_event(
        self,
        session_id: str,
        event_type: EventType,
        phase: str | None,
        payload: dict[str, Any],
    ) -> None:
        """Create and persist an event via the repository."""
        event = create_event(session_id, self.name, event_type, phase, payload)
        self._repo.emit_event(event)

    # ------------------------------------------------------------------
    # Note writing
    # ------------------------------------------------------------------

    def write_note(
        self,
        session_id: str,
        content: str,
        note_type: NoteType = NoteType.observation,
        references: list[dict[str, Any]] | None = None,
    ) -> AgentNote:
        """Create and persist an agent note, returning it."""
        note = AgentNote(
            id=str(uuid.uuid4()),
            session_id=session_id,
            agent_name=self.name,
            note_type=note_type,
            content=content,
            note_references=references or [],
        )
        self._repo.insert_agent_note(note)
        return note

    # ------------------------------------------------------------------
    # Memory recall
    # ------------------------------------------------------------------

    def recall_memories(
        self,
        query: str,
        type: Any = None,
        scope: Any = None,
        limit: int = 5,
    ) -> list:
        """Delegate memory recall to the MemoryManager."""
        return self._memory_mgr.recall(query, type=type, scope=scope, limit=limit)
