"""JuryAgent: score completed missions and rate strategies."""
from __future__ import annotations

import uuid
from typing import Any

from ss.agents.base import BaseAgent
from ss.blackboard.models import AgentName, StrategyScore
from ss.config import Config
from ss.pipeline.events import EventType


SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "completeness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "elegance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "robustness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "efficiency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
    },
    "required": [
        "correctness", "completeness", "elegance", "robustness", "efficiency", "reasoning"
    ],
}


class JuryAgent(BaseAgent):
    """Scores mission results and assigns a rating label to strategies."""

    def __init__(
        self,
        repo: Any,
        sampling: Any,
        memory_mgr: Any,
        vector_store: Any,
        config: Config,
    ) -> None:
        super().__init__(repo, sampling, memory_mgr, vector_store)
        self._config = config

    @property
    def name(self) -> AgentName:
        return AgentName.jury

    @property
    def persona(self) -> str:
        return (
            "You are the Jury Agent. Your role is to objectively score the outcomes "
            "of mission executions across multiple dimensions: correctness, completeness, "
            "elegance, robustness, and efficiency. Score each dimension from 0.0 to 1.0 "
            "and provide clear reasoning for your scores."
        )

    @property
    def temperature(self) -> float:
        return 0.2

    async def execute(
        self,
        session_id: str,
        *,
        missions: list,
        **kwargs: Any,
    ) -> list[StrategyScore]:
        """Score all succeeded missions.

        Args:
            session_id: The active session id.
            missions: List of Mission objects to evaluate.

        Returns:
            List of StrategyScore objects for succeeded missions.
        """
        scores: list[StrategyScore] = []

        for mission in missions:
            if getattr(mission, "status", "") != "succeeded":
                continue

            # Get strategy and results
            strategy = self._repo.get_strategy(mission.strategy_id)
            if strategy is None:
                continue
            results = self._repo.list_mission_results(mission.id)

            # Build results summary
            results_text = "\n".join(
                f"  Step {r.step_index}: {r.action}\n"
                f"    Expected: {r.expected_outcome}\n"
                f"    Actual: {r.actual_outcome}\n"
                f"    Success: {r.success}"
                for r in results
            )

            system_prompt = (
                "Score this mission execution across five dimensions: correctness, "
                "completeness, elegance, robustness, and efficiency. Each score is "
                "0.0 (worst) to 1.0 (best). Provide clear reasoning."
            )
            user_message = (
                f"Strategy: {strategy.description}\n"
                f"Objective: {strategy.objective}\n\n"
                f"Mission results ({len(results)} steps):\n{results_text}"
            )

            score_data = await self.llm_call_structured(
                system_prompt, user_message, SCORING_SCHEMA
            )

            correctness = float(score_data.get("correctness", 0.0))
            completeness = float(score_data.get("completeness", 0.0))
            elegance = float(score_data.get("elegance", 0.0))
            robustness = float(score_data.get("robustness", 0.0))
            efficiency = float(score_data.get("efficiency", 0.0))
            reasoning = score_data.get("reasoning", "")

            # Calculate weighted total
            weights = self._config.score_weights
            weighted_total = (
                correctness * weights.get("correctness", 0.30)
                + completeness * weights.get("completeness", 0.25)
                + robustness * weights.get("robustness", 0.20)
                + elegance * weights.get("elegance", 0.15)
                + efficiency * weights.get("efficiency", 0.10)
            )

            strategy_score = StrategyScore(
                id=str(uuid.uuid4()),
                strategy_id=mission.strategy_id,
                session_id=session_id,
                correctness=correctness,
                completeness=completeness,
                elegance=elegance,
                robustness=robustness,
                efficiency=efficiency,
                weighted_total=weighted_total,
                reasoning=reasoning,
            )
            self._repo.insert_strategy_score(strategy_score)

            # Determine rating label based on thresholds
            if weighted_total >= self._config.proven_threshold:
                rating_label = "proven"
            elif weighted_total >= self._config.archive_threshold:
                rating_label = "adequate"
            else:
                rating_label = "failed"

            # Update strategy with score and rating
            self._repo.update_strategy_status(
                mission.strategy_id,
                status="completed",
                rating_label=rating_label,
                jury_score=weighted_total,
            )

            self.emit_event(
                session_id,
                EventType.JURY_SCORED,
                "score",
                {
                    "strategy_id": mission.strategy_id,
                    "mission_id": mission.id,
                    "weighted_total": weighted_total,
                    "rating_label": rating_label,
                    "correctness": correctness,
                    "completeness": completeness,
                    "elegance": elegance,
                    "robustness": robustness,
                    "efficiency": efficiency,
                },
            )

            scores.append(strategy_score)

        return scores
