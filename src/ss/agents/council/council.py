"""LLMCouncil: multi-model propose → anonymised review → chairman synthesis."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ss.agents.council.chairman import Chairman
from ss.agents.council.member import AnonymisedProposal, CouncilMember
from ss.agents.council.personas import DEFAULT_PERSONAS, personas_for
from ss.blackboard.models import AgentName, MemoryType, Strategy
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.pipeline.events import EventType, create_event

logger = logging.getLogger(__name__)

_LABELS = [f"Voice {chr(ord('A') + i)}" for i in range(26)]


class LLMCouncil:
    """Drop-in replacement for StrategistAgent: a persona-driven multi-model council."""

    def __init__(
        self,
        repo: Repository,
        bundle: Any,
        memory_mgr: Any,
        vector_store: Any,
        config: Config,
        pml: Any = None,
    ) -> None:
        self._repo = repo
        self._bundle = bundle
        self._memory_mgr = memory_mgr
        self._vector_store = vector_store
        self._config = config
        self._pml = pml

    def _recall_texts(self, query: str, mem_type: MemoryType, limit: int = 3) -> list[str]:
        return [m.content for m in self._memory_mgr.recall(query, type=mem_type, limit=limit)]

    def _converged(self, reviews: list[str]) -> bool:
        """True if the round's reviews agree (mean pairwise cosine >= threshold).

        Best-effort: any embedding failure returns False (never stop early on error).
        """
        if len(reviews) < 2:
            return True
        try:
            import numpy as np

            encoder = self._vector_store.encoder
            vecs = [encoder.encode(r) for r in reviews]
            sims: list[float] = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    a, b = vecs[i], vecs[j]
                    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
                    sims.append(float(np.dot(a, b)) / denom)
            return (sum(sims) / len(sims)) >= self._config.council_convergence_threshold
        except Exception:
            return False

    async def execute(
        self,
        session_id: str,
        *,
        issue: Any,
        num_strategies: int = 3,
        failure_context: str | None = None,
    ) -> list[Strategy]:
        """Run the three council stages and persist N ranked strategies."""
        anti_patterns = self._recall_texts(issue.summary, MemoryType.anti_pattern)
        anti_patterns += self._recall_texts(issue.summary, MemoryType.bad_practice)
        if failure_context:
            anti_patterns.append(f"Previous round failed: {failure_context}")

        # Resolve optional PML project context (injected into proposals + reviews
        # when a ProjectMemory instance is supplied; skipped when pml is None).
        project_context = (
            self._pml.as_context(["INTENT", "DECISION", "TIMELINE"])
            if self._pml is not None
            else ""
        )

        # One council member per persona (thinking lenses); models are
        # cycled across personas, so even a single configured model still yields
        # the full set of thinking lenses. model=None lets the adapter use its
        # bound default (and is ignored entirely in MCP sampling mode).
        member_models = self._bundle.council_member_models() or [None]
        personas = personas_for(self._config.council_personas) or DEFAULT_PERSONAS
        members = [
            CouncilMember(
                self._bundle,
                persona=p,
                model=member_models[i % len(member_models)],
                temperature=self._config.council_persona_temps.get(p.key),
            )
            for i, p in enumerate(personas)
        ]

        # Stage 1 — independent proposals (parallel)
        proposals = await asyncio.gather(
            *[m.propose(issue, anti_patterns, project_context=project_context) for m in members]
        )

        # Anonymise: rotate so no member reviews in original order; relabel A/B/C...
        anonymised = [
            AnonymisedProposal(label=_LABELS[i], content=p)
            for i, p in enumerate(proposals)
        ]
        if self._config.council_anonymise_identities and len(anonymised) > 1:
            anonymised = anonymised[1:] + anonymised[:1]

        # Stage 2 — anonymised peer review (parallel). Run up to
        # `council_review_rounds`, but STOP EARLY once the members' reviews
        # converge (mean pairwise similarity >= threshold). Keep only the latest
        # round's reviews: they reflect the converged consensus and keep the
        # chairman's prompt bounded.
        reviews: list[str] = []
        rounds = max(1, self._config.council_review_rounds)
        for round_idx in range(rounds):
            reviews = list(await asyncio.gather(
                *[m.review(anonymised, anti_patterns, project_context=project_context)
                  for m in members]
            ))
            if round_idx + 1 >= rounds or self._converged(reviews):
                break

        # Stage 3 — chairman synthesis
        chairman = Chairman(
            self._bundle,
            model=self._bundle.council_chairman_model(),
            num_strategies=num_strategies,
        )
        strategies = await chairman.synthesise(
            session_id, issue, list(proposals), reviews, anti_patterns
        )

        for strategy in strategies:
            self._repo.insert_strategy(strategy)
            self._vector_store.index("strategy", strategy.id, strategy.description)

        event = create_event(
            session_id,
            AgentName.strategist,  # keep event schema/agent stable
            EventType.STRATEGIES_GENERATED,
            "generate",
            {"count": len(strategies), "strategy_ids": [s.id for s in strategies]},
        )
        self._repo.emit_event(event)

        return strategies
