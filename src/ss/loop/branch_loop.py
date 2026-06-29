"""BranchLoop: wraps Taktik → Judge → Mission in a bounded agentic loop."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ss.agents.judge import JudgeAgent
from ss.agents.mission import MissionAgent
from ss.agents.taktik_planner import TaktikPlannerAgent
from ss.blackboard.models import Mission
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.loop.postcondition import PostconditionChecker
from ss.loop.progress import ProgressTracker
from ss.pipeline.events import EventType, create_event

logger = logging.getLogger(__name__)


@dataclass
class BranchResult:
    """Terminal outcome of a strategy branch."""
    strategy_id: str
    status: str  # "succeeded" | "failed"
    mission: Mission | None
    iterations: int
    reason: str
    required_skills: list[str] = field(default_factory=list)


class BranchLoop:
    """Runs one strategy through a bounded observe→plan→verify→act→check loop."""

    def __init__(
        self,
        repo: Repository,
        bundle: Any,
        memory_mgr: Any,
        vector_store: Any,
        config: Config,
        skill_resolver: Any = None,
        pml: Any = None,
        cil: Any = None,
    ) -> None:
        self._repo = repo
        self._config = config
        self._pml = pml
        self._cil = cil
        self._taktik_planner = TaktikPlannerAgent(
            repo, bundle.for_agent("taktik_planner"), memory_mgr, vector_store,
            skill_resolver=skill_resolver, cil=cil, pml=pml,
        )
        self._judge = JudgeAgent(repo, bundle.for_agent("judge"), memory_mgr, vector_store, pml=pml)
        self._mission = MissionAgent(repo, bundle.for_agent("mission"), memory_mgr, vector_store, cil=cil)
        self._checker = PostconditionChecker(bundle.for_agent("judge"), cil=cil)

    def _issue_for(self, session_id: str):
        issues = self._repo.get_issue_by_session(session_id)
        return issues[0] if issues else None

    async def _verified_taktik(self, session_id: str, strategy: Any, issue: Any):
        """Inner judge-retry loop: return (taktik, last_rejection_or_None)."""
        rejection: str | None = None
        for attempt in range(1, self._config.max_judge_retries + 1):
            taktik = await self._taktik_planner.execute(
                session_id, strategy=strategy, attempt=attempt, rejection_reason=rejection
            )
            verification = await self._judge.execute(
                session_id, taktik=taktik, strategy=strategy, issue=issue
            )
            if verification.get("verified"):
                return taktik, None
            issues_found = verification.get("issues_found", [])
            rejection = "; ".join(issues_found) if issues_found else verification.get("reasoning", "")
            self._repo.emit_event(create_event(
                session_id, self._taktik_planner.name, EventType.JUDGE_REJECTED_LOOP, "retry",
                {"strategy_id": strategy.id, "taktik_id": taktik.id, "attempt": attempt,
                 "rejection_reason": rejection,
                 "remaining_attempts": self._config.max_judge_retries - attempt},
            ))
        return None, rejection

    async def run(self, session_id: str, strategy: Any) -> BranchResult:
        """Execute the strategy branch; return a terminal BranchResult (never raises)."""
        issue = self._issue_for(session_id)
        postcondition = getattr(issue, "postcondition", "") if issue else ""
        tracker = ProgressTracker(self._config.max_loop_iterations, self._config.no_progress_threshold)
        last_mission: Mission | None = None
        last_skills: list[str] = []

        while True:
            # OBSERVE
            # (state is read fresh from the blackboard each iteration via the repo)
            if self._cil is not None:
                try:
                    self._cil.session()
                except Exception:
                    logger.warning("CIL session() failed during OBSERVE; continuing", exc_info=True)

            # PLAN + VERIFY
            taktik, rejection = await self._verified_taktik(session_id, strategy, issue)
            if taktik is None:
                return BranchResult(strategy.id, "failed", last_mission, tracker.iterations,
                                    f"no verified plan: {rejection}", last_skills)
            last_skills = list(taktik.required_skills)

            # ACT
            mission = await self._mission.execute(
                session_id, taktik=taktik, strategy_id=strategy.id
            )
            last_mission = mission

            # CHECK
            results = self._repo.list_mission_results(mission.id)
            check = await self._checker.check(postcondition, {"mission_results": results})
            tracker.record(check)

            # DECIDE
            if check.passed:
                return BranchResult(strategy.id, "succeeded", mission, tracker.iterations,
                                    "postcondition satisfied", last_skills)
            if tracker.budget_exhausted():
                return BranchResult(strategy.id, "failed", mission, tracker.iterations,
                                    "iteration budget exhausted", last_skills)
            if tracker.is_stagnant():
                return BranchResult(strategy.id, "failed", mission, tracker.iterations,
                                    "no progress - stagnation, escalate to re-strategise", last_skills)
