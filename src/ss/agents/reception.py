"""ReceptionAgent: intake and classify incoming problems."""
from __future__ import annotations

import uuid
from typing import Any

from ss.agents.base import BaseAgent
from ss.blackboard.models import AgentName, Issue, IssueClassification
from ss.pipeline.events import EventType


ISSUE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "classification": {
            "type": "string",
            "enum": [
                "bug", "architecture", "performance", "refactor",
                "security", "testing", "deployment", "documentation", "unknown",
            ],
        },
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "who": {"type": "string"},
        "where_location": {"type": "string"},
        "why_reason": {"type": "string"},
        "precondition": {"type": "string"},
        "postcondition": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary", "classification", "severity", "who", "where_location",
        "why_reason", "precondition", "postcondition", "key_points",
    ],
}


class ReceptionAgent(BaseAgent):
    """Receives a problem statement, extracts structured issue information."""

    def __init__(self, repo, sampling, memory_mgr, vector_store, pml=None):
        """Initialize ReceptionAgent with optional ProjectMemory Layer (PML).

        Args:
            repo: The repository for database access.
            sampling: The sampling adapter for LLM calls.
            memory_mgr: The memory manager for recall.
            vector_store: The vector store for indexing.
            pml: Optional ProjectMemory instance for grounding with STATE/HEALTH context.
        """
        super().__init__(repo, sampling, memory_mgr, vector_store)
        self._pml = pml

    @property
    def name(self) -> AgentName:
        return AgentName.reception

    @property
    def persona(self) -> str:
        return (
            "You are the Reception Agent. Your role is to carefully intake a software "
            "engineering problem, classify it, and extract structured information about "
            "who is affected, where it occurs, why it happens, and what the expected "
            "outcome should be. Be precise and analytical."
        )

    @property
    def temperature(self) -> float:
        return 0.1

    async def execute(
        self,
        session_id: str,
        *,
        problem: str,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Issue:
        """Intake a problem, classify it, and create an Issue record.

        Args:
            session_id: The active session id.
            problem: The raw problem text to analyze.
            context: Optional additional context dict.

        Returns:
            The created and persisted Issue.
        """
        self.emit_event(session_id, EventType.AGENT_STARTED, "start", {"agent": self.name})

        # Recall relevant client context memories
        memories = self.recall_memories(problem, limit=3)
        memory_context = ""
        if memories:
            memory_context = "\n\nRelevant context from memory:\n" + "\n".join(
                f"- {m.content}" for m in memories
            )

        system_prompt = (
            "Analyze the following software engineering problem and extract structured "
            "information. Return a JSON object matching the required schema."
        )
        user_message = f"Problem:\n{problem}"
        if context:
            user_message += f"\n\nAdditional context:\n{context}"
        if memory_context:
            user_message += memory_context

        # Prepend project state/health context if PML is available
        if self._pml is not None:
            pml_context = self._pml.as_context(["STATE", "HEALTH"])
            if pml_context:
                user_message = f"{pml_context}\n\n{user_message}"

        data = await self.llm_call_structured(system_prompt, user_message, ISSUE_SCHEMA)

        issue = Issue(
            id=str(uuid.uuid4()),
            session_id=session_id,
            summary=data["summary"],
            classification=IssueClassification(data["classification"]),
            severity=data.get("severity", "medium"),
            who=data["who"],
            where_location=data["where_location"],
            why_reason=data["why_reason"],
            precondition=data["precondition"],
            postcondition=data["postcondition"],
            key_points=data["key_points"],
        )

        self._repo.insert_issue(issue)
        self._vector_store.index("issue", issue.id, issue.summary)

        self.emit_event(
            session_id,
            EventType.RECEPTION_INTAKE,
            "intake",
            {
                "issue_id": issue.id,
                "classification": str(issue.classification),
                "severity": issue.severity,
                "summary": issue.summary,
            },
        )

        return issue
