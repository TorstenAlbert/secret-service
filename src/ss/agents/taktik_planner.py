"""TaktikPlannerAgent: create a detailed step-by-step plan for a strategy."""
from __future__ import annotations

import uuid
from typing import Any

from ss.agents.base import BaseAgent
from ss.blackboard.models import AgentName, MemoryType, Taktik, TaktikStep
from ss.pipeline.events import EventType


TAKTIK_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "instruction": {"type": "string"},
                    "expected_outcome": {"type": "string"},
                },
                "required": ["index", "instruction", "expected_outcome"],
            },
        },
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "estimated_complexity": {
            "type": "string",
            "enum": ["low", "medium", "high", "very_high"],
        },
    },
    "required": ["steps", "required_skills", "estimated_complexity"],
}


class TaktikPlannerAgent(BaseAgent):
    """Creates a concrete execution plan (taktik) for a given strategy."""

    @property
    def name(self) -> AgentName:
        return AgentName.taktik_planner

    @property
    def persona(self) -> str:
        return (
            "You are the Taktik Planner Agent. Your role is to create detailed, "
            "concrete, step-by-step execution plans for software engineering strategies. "
            "Each step must have a clear instruction and measurable expected outcome. "
            "Be specific and actionable."
        )

    @property
    def temperature(self) -> float:
        return 0.8

    async def execute(
        self,
        session_id: str,
        *,
        strategy: Any,
        attempt: int = 1,
        rejection_reason: str | None = None,
        **kwargs: Any,
    ) -> Taktik:
        """Plan a taktik for the given strategy.

        Args:
            session_id: The active session id.
            strategy: The Strategy object to plan for.
            attempt: Attempt number (used for retry logic).
            rejection_reason: Reason the previous taktik was rejected (if retrying).

        Returns:
            The created and persisted Taktik.
        """
        # Recall short-term knowledge memories for relevant context
        memories = self.recall_memories(
            strategy.description, type=MemoryType.knowledge, limit=3
        )
        memory_ctx = ""
        if memories:
            memory_ctx = "\n\nRecent knowledge:\n" + "\n".join(
                f"- {m.content}" for m in memories
            )

        rejection_ctx = ""
        if rejection_reason:
            rejection_ctx = (
                f"\n\nIMPORTANT - Previous plan was rejected for this reason:\n"
                f"{rejection_reason}\n"
                "Please address this issue in your new plan."
            )

        system_prompt = (
            "Create a detailed, concrete step-by-step execution plan for the given "
            "software engineering strategy. Include all required skills and estimate "
            "the complexity. Return as a JSON object."
        )
        user_message = (
            f"Strategy description: {strategy.description}\n"
            f"Objective: {strategy.objective}\n"
            f"Approach type: {strategy.approach_type}\n"
            f"Attempt #{attempt}"
        ) + memory_ctx + rejection_ctx

        data = await self.llm_call_structured(system_prompt, user_message, TAKTIK_SCHEMA)

        steps = [
            TaktikStep(
                index=s["index"],
                instruction=s["instruction"],
                expected_outcome=s["expected_outcome"],
            )
            for s in data.get("steps", [])
        ]

        taktik = Taktik(
            id=str(uuid.uuid4()),
            strategy_id=strategy.id,
            session_id=session_id,
            steps=steps,
            required_skills=data.get("required_skills", []),
            estimated_complexity=data.get("estimated_complexity"),
            attempt_number=attempt,
        )

        self._repo.insert_taktik(taktik)
        taktik_text = strategy.description + " " + " ".join(
            s.instruction for s in steps
        )
        self._vector_store.index("taktik", taktik.id, taktik_text)

        self.emit_event(
            session_id,
            EventType.TAKTIK_PLANNED,
            "plan",
            {
                "taktik_id": taktik.id,
                "strategy_id": strategy.id,
                "step_count": len(steps),
                "attempt": attempt,
            },
        )

        return taktik
