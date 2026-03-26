"""MasterAgent: orchestrates the pipeline and distributes learnings."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from ss.agents.base import BaseAgent
from ss.blackboard.models import (
    AgentName,
    Memory,
    MemoryScope,
    MemoryType,
    NoteType,
)
from ss.pipeline.events import EventType


CLIENT_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "expertise_level": {
            "type": "string",
            "enum": ["beginner", "intermediate", "advanced", "expert"],
        },
        "domains": {"type": "array", "items": {"type": "string"}},
        "communication_style": {
            "type": "string",
            "enum": ["technical", "business", "mixed"],
        },
    },
    "required": ["expertise_level", "domains", "communication_style"],
}

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "key_insights": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["final_answer", "key_insights", "recommendations"],
}


class MasterAgent(BaseAgent):
    """The Master orchestrates session flow and synthesizes outcomes."""

    @property
    def name(self) -> AgentName:
        return AgentName.master

    @property
    def persona(self) -> str:
        return (
            "You are the Master Agent. You orchestrate the problem-solving pipeline, "
            "assess client needs, synthesize final answers, and distribute learnings "
            "across memory for future sessions. You think strategically and holistically."
        )

    @property
    def temperature(self) -> float:
        return 0.3

    async def execute(self, session_id: str, **kwargs: Any) -> Any:
        """Dispatch to sub-method based on kwargs.

        Modes:
        - join_session(session_id, issue_summary, client_id)
        - synthesize(session_id, winning_strategy, score, mission_results)
        - distribute_learnings(session_id, mission, strategy, score)
        """
        if "issue_summary" in kwargs and "client_id" in kwargs:
            return await self.join_session(
                session_id,
                issue_summary=kwargs["issue_summary"],
                client_id=kwargs["client_id"],
            )
        elif "winning_strategy" in kwargs:
            return await self.synthesize(
                session_id,
                winning_strategy=kwargs["winning_strategy"],
                score=kwargs["score"],
                mission_results=kwargs.get("mission_results", []),
            )
        elif "mission" in kwargs and "strategy" in kwargs:
            return await self.distribute_learnings(
                session_id,
                mission=kwargs["mission"],
                strategy=kwargs["strategy"],
                score=kwargs["score"],
            )
        else:
            raise ValueError(f"Unknown execute mode for MasterAgent. kwargs: {list(kwargs)}")

    async def join_session(
        self,
        session_id: str,
        *,
        issue_summary: str,
        client_id: str,
    ) -> dict:
        """Assess the client and emit MASTER_JOINED."""
        system_prompt = (
            "Assess the client's expertise level, relevant domains, and preferred "
            "communication style based on the issue they submitted."
        )
        user_message = (
            f"Client ID: {client_id}\n"
            f"Issue summary: {issue_summary}\n\n"
            "Return a structured JSON assessment of this client."
        )

        assessment = await self.llm_call_structured(
            system_prompt, user_message, CLIENT_ASSESSMENT_SCHEMA
        )

        self.emit_event(
            session_id,
            EventType.MASTER_JOINED,
            "join",
            {
                "client_id": client_id,
                "assessment": assessment,
            },
        )

        return assessment

    async def synthesize(
        self,
        session_id: str,
        *,
        winning_strategy: Any,
        score: float,
        mission_results: list,
    ) -> dict:
        """Gather notes, synthesize final answer, emit MASTER_SYNTHESIZED."""
        notes = self._repo.list_agent_notes(session_id)
        notes_text = "\n".join(
            f"[{n.agent_name} / {n.note_type}]: {n.content}" for n in notes
        )

        system_prompt = (
            "You are synthesizing the final answer for a software engineering problem. "
            "Review all agent notes, the winning strategy, and mission results to produce "
            "a clear, actionable final answer."
        )
        user_message = (
            f"Winning strategy: {winning_strategy.description if hasattr(winning_strategy, 'description') else winning_strategy}\n"
            f"Jury score: {score}\n\n"
            f"Agent notes:\n{notes_text}\n\n"
            f"Mission results: {len(mission_results)} results"
        )

        synthesis = await self.llm_call_structured(
            system_prompt, user_message, SYNTHESIS_SCHEMA
        )

        self.emit_event(
            session_id,
            EventType.MASTER_SYNTHESIZED,
            "synthesize",
            {
                "strategy_id": getattr(winning_strategy, "id", str(winning_strategy)),
                "score": score,
                "final_answer_preview": synthesis.get("final_answer", "")[:200],
            },
        )

        return synthesis

    async def distribute_learnings(
        self,
        session_id: str,
        *,
        mission: Any,
        strategy: Any,
        score: float,
    ) -> None:
        """Create memory records based on mission outcome."""
        succeeded = getattr(mission, "status", "") == "succeeded"
        strategy_desc = getattr(strategy, "description", str(strategy))
        strategy_id = getattr(strategy, "id", str(strategy))

        now = datetime.now(timezone.utc)

        if succeeded:
            # Good practice memory (long_term)
            self._memory_mgr.store(
                Memory(
                    id=str(uuid.uuid4()),
                    type=MemoryType.good_practice,
                    scope=MemoryScope.long_term,
                    source_session_id=session_id,
                    source_agent=self.name,
                    content=f"Successful strategy: {strategy_desc}",
                    structured_content={"strategy_id": strategy_id, "score": score},
                    confidence=min(1.0, score),
                )
            )
            # Pattern memory (permanent if score >= 0.75)
            if score >= 0.75:
                self._memory_mgr.store(
                    Memory(
                        id=str(uuid.uuid4()),
                        type=MemoryType.pattern,
                        scope=MemoryScope.permanent,
                        source_session_id=session_id,
                        source_agent=self.name,
                        content=f"Proven pattern (score={score:.2f}): {strategy_desc}",
                        structured_content={"strategy_id": strategy_id, "score": score},
                        confidence=score,
                    )
                )
        else:
            # Bad practice memory (long_term)
            self._memory_mgr.store(
                Memory(
                    id=str(uuid.uuid4()),
                    type=MemoryType.bad_practice,
                    scope=MemoryScope.long_term,
                    source_session_id=session_id,
                    source_agent=self.name,
                    content=f"Failed strategy: {strategy_desc}",
                    structured_content={"strategy_id": strategy_id, "score": score},
                    confidence=1.0,
                )
            )
            # Anti-pattern memory (permanent if score < 0.3)
            if score < 0.3:
                self._memory_mgr.store(
                    Memory(
                        id=str(uuid.uuid4()),
                        type=MemoryType.anti_pattern,
                        scope=MemoryScope.permanent,
                        source_session_id=session_id,
                        source_agent=self.name,
                        content=f"Anti-pattern (score={score:.2f}): {strategy_desc}",
                        structured_content={"strategy_id": strategy_id, "score": score},
                        confidence=1.0,
                    )
                )

        # Knowledge memory (short_term, expires in 1 hour)
        self._memory_mgr.store(
            Memory(
                id=str(uuid.uuid4()),
                type=MemoryType.knowledge,
                scope=MemoryScope.short_term,
                source_session_id=session_id,
                source_agent=self.name,
                content=f"Session learning: strategy '{strategy_desc}' outcome={'success' if succeeded else 'failure'} score={score:.2f}",
                structured_content={
                    "strategy_id": strategy_id,
                    "score": score,
                    "succeeded": succeeded,
                },
                expires_at=now + timedelta(hours=1),
            )
        )
