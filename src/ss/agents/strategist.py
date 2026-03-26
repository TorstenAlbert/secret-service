"""StrategistAgent: generate multiple solution strategies for an issue."""
from __future__ import annotations

import uuid
from typing import Any

from ss.agents.base import BaseAgent
from ss.blackboard.models import AgentName, MemoryType, Strategy
from ss.pipeline.events import EventType


STRATEGIES_SCHEMA = {
    "type": "object",
    "properties": {
        "strategies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "objective": {"type": "string"},
                    "approach_type": {"type": "string"},
                    "confidence": {"type": "number"},
                    "rank": {"type": "integer"},
                },
                "required": ["description", "objective", "rank"],
            },
        }
    },
    "required": ["strategies"],
}


class StrategistAgent(BaseAgent):
    """Generates N ranked strategies to solve a given issue."""

    @property
    def name(self) -> AgentName:
        return AgentName.strategist

    @property
    def persona(self) -> str:
        return (
            "You are the Strategist Agent. Your role is to generate multiple distinct "
            "solution strategies for software engineering problems. Each strategy should "
            "be viable, clearly described, and ranked by confidence. Think creatively "
            "and consider multiple angles of attack."
        )

    @property
    def temperature(self) -> float:
        return 0.9

    async def execute(
        self,
        session_id: str,
        *,
        issue: Any,
        num_strategies: int = 3,
        failure_context: str | None = None,
        **kwargs: Any,
    ) -> list[Strategy]:
        """Generate N strategies for the given issue.

        Args:
            session_id: The active session id.
            issue: The Issue object to generate strategies for.
            num_strategies: Number of strategies to generate.
            failure_context: Optional context about previous failures.

        Returns:
            List of created and persisted Strategy objects.
        """
        # Recall relevant memories
        good_practices = self.recall_memories(
            issue.summary, type=MemoryType.good_practice, limit=3
        )
        bad_practices = self.recall_memories(
            issue.summary, type=MemoryType.bad_practice, limit=3
        )

        good_ctx = ""
        if good_practices:
            good_ctx = "\n\nSuccessful approaches from memory:\n" + "\n".join(
                f"- {m.content}" for m in good_practices
            )
        bad_ctx = ""
        if bad_practices:
            bad_ctx = "\n\nApproaches that have failed:\n" + "\n".join(
                f"- {m.content}" for m in bad_practices
            )
        failure_ctx = ""
        if failure_context:
            failure_ctx = f"\n\nPrevious failure context:\n{failure_context}"

        system_prompt = (
            f"Generate exactly {num_strategies} distinct strategies to solve the given "
            "software engineering issue. Each strategy must be different in approach. "
            "Return them as a JSON object with a 'strategies' array."
        )
        issue_text = (
            f"Issue summary: {issue.summary}\n"
            f"Classification: {issue.classification}\n"
            f"Severity: {issue.severity}\n"
            f"Who: {issue.who}\n"
            f"Where: {issue.where_location}\n"
            f"Why: {issue.why_reason}\n"
            f"Precondition: {issue.precondition}\n"
            f"Postcondition: {issue.postcondition}\n"
            f"Key points: {', '.join(issue.key_points)}"
        )
        user_message = issue_text + good_ctx + bad_ctx + failure_ctx

        data = await self.llm_call_structured(system_prompt, user_message, STRATEGIES_SCHEMA)

        strategies: list[Strategy] = []
        for i, s in enumerate(data.get("strategies", [])):
            strategy = Strategy(
                id=str(uuid.uuid4()),
                session_id=session_id,
                description=s["description"],
                objective=s["objective"],
                approach_type=s.get("approach_type"),
                rank=s.get("rank", i + 1),
                confidence=s.get("confidence"),
            )
            self._repo.insert_strategy(strategy)
            self._vector_store.index("strategy", strategy.id, strategy.description)
            strategies.append(strategy)

        self.emit_event(
            session_id,
            EventType.STRATEGIES_GENERATED,
            "generate",
            {
                "count": len(strategies),
                "strategy_ids": [s.id for s in strategies],
            },
        )

        return strategies
