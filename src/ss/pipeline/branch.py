"""StrategyBranch: run one strategy through Taktik Planner → Judge → Mission."""
from __future__ import annotations

from typing import Any

from ss.agents.judge import JudgeAgent
from ss.agents.mission import MissionAgent
from ss.agents.taktik_planner import TaktikPlannerAgent
from ss.blackboard.models import Mission
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.pipeline.events import EventType, create_event


class StrategyExhausted(Exception):
    """Raised when all judge retries for a strategy have been exhausted."""

    def __init__(self, strategy_id: str, reason: str) -> None:
        self.strategy_id = strategy_id
        self.reason = reason
        super().__init__(f"Strategy {strategy_id} exhausted: {reason}")


class StrategyBranch:
    """Executes one strategy through: Taktik Planner → Judge → Mission.

    Handles the judge retry loop internally. Raises StrategyExhausted if all
    retries are consumed without a verified taktik.
    """

    def __init__(
        self,
        repo: Repository,
        sampling: Any,
        memory_mgr: Any,
        vector_store: Any,
        config: Config,
    ) -> None:
        self._repo = repo
        self._config = config
        self._taktik_planner = TaktikPlannerAgent(repo, sampling, memory_mgr, vector_store)
        self._judge = JudgeAgent(repo, sampling, memory_mgr, vector_store)
        self._mission = MissionAgent(repo, sampling, memory_mgr, vector_store)

    async def run(self, session_id: str, strategy: Any) -> Mission:
        """Execute the strategy through the full taktik → judge → mission pipeline.

        Args:
            session_id: The active session id.
            strategy: The Strategy object to execute.

        Returns:
            The completed Mission object.

        Raises:
            StrategyExhausted: If all judge retries are consumed without verification.
        """
        issues = self._repo.get_issue_by_session(session_id)
        # Use the first issue found (most recently inserted)
        issue = issues[0] if issues else None

        rejection_reason: str | None = None
        last_rejection: str = "Unknown rejection"

        for attempt in range(1, self._config.max_judge_retries + 1):
            taktik = await self._taktik_planner.execute(
                session_id,
                strategy=strategy,
                attempt=attempt,
                rejection_reason=rejection_reason,
            )

            verification = await self._judge.execute(
                session_id,
                taktik=taktik,
                strategy=strategy,
                issue=issue,
            )

            verified = bool(verification.get("verified", False))

            if verified:
                mission = await self._mission.execute(
                    session_id,
                    taktik=taktik,
                    strategy_id=strategy.id,
                )
                return mission

            # Not verified: emit rejection event and prepare for next attempt
            issues_found = verification.get("issues_found", [])
            reasoning = verification.get("reasoning", "")
            rejection_reason = "; ".join(issues_found) if issues_found else reasoning
            last_rejection = rejection_reason

            event = create_event(
                session_id,
                self._taktik_planner.name,
                EventType.JUDGE_REJECTED_LOOP,
                "retry",
                {
                    "strategy_id": strategy.id,
                    "taktik_id": taktik.id,
                    "attempt": attempt,
                    "rejection_reason": rejection_reason,
                    "remaining_attempts": self._config.max_judge_retries - attempt,
                },
            )
            self._repo.emit_event(event)

        raise StrategyExhausted(strategy.id, last_rejection)
