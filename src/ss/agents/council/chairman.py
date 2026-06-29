"""Chairman: synthesises council proposals + reviews into ranked strategies."""
from __future__ import annotations

import json
import uuid
from typing import Any

from ss.agents.council.member import _anti_pattern_block
from ss.agents.council.personas import CHAIRMAN
from ss.agents.strategist import STRATEGIES_SCHEMA
from ss.blackboard.models import Strategy
from ss.sampling.adapter import SamplingAdapter


class Chairman:
    """Synthesises the council's debate into N ranked, distinct strategies."""

    def __init__(self, bundle: Any, model: str, num_strategies: int, temperature: float = 0.4) -> None:
        """Bind the council bundle, chairman model, and target strategy count."""
        self._bundle = bundle
        self._model = model
        self._num_strategies = num_strategies
        self._temperature = temperature

    async def synthesise(
        self,
        session_id: str,
        issue: Any,
        proposals: list[str],
        reviews: list[str],
        anti_patterns: list[str],
    ) -> list[Strategy]:
        """Produce N ranked Strategy objects (not yet persisted)."""
        system = (
            f"You are {CHAIRMAN.name}. {CHAIRMAN.lens}\n\n"
            f"Synthesise exactly {self._num_strategies} "
            "distinct, competing strategies from the members' proposals and reviews. "
            "Favour ideas the reviews rated highly; discard ideas that match anti-patterns. "
            "Return a JSON object with a 'strategies' array."
            "\n\nThe JSON must conform to this schema:\n"
            f"{json.dumps(STRATEGIES_SCHEMA, indent=2)}\n"
            "Output ONLY the JSON object, optionally wrapped in ```json ... ``` fences."
        )
        proposals_text = "\n\n".join(f"Proposal {i+1}:\n{p}" for i, p in enumerate(proposals))
        reviews_text = "\n\n".join(f"Review {i+1}:\n{r}" for i, r in enumerate(reviews))
        user = (
            f"Issue: {getattr(issue, 'summary', issue)}\n"
            f"Postcondition: {getattr(issue, 'postcondition', '')}"
            f"{_anti_pattern_block(anti_patterns)}"
            f"\n## MEMBER PROPOSALS\n{proposals_text}"
            f"\n\n## PEER REVIEWS\n{reviews_text}"
        )
        text = await self._bundle.council_complete(
            model=self._model,
            system_prompt=system,
            messages=[{"role": "user", "content": user}],
            temperature=self._temperature,
        )
        data = SamplingAdapter._parse_json(text)

        strategies: list[Strategy] = []
        for i, s in enumerate(data.get("strategies", [])[: self._num_strategies]):
            strategies.append(
                Strategy(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    description=s["description"],
                    objective=s["objective"],
                    approach_type=s.get("approach_type"),
                    rank=s.get("rank", i + 1),
                    confidence=s.get("confidence"),
                )
            )
        return strategies
