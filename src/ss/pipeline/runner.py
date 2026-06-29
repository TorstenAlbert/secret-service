"""SessionRunner: orchestrates the full multi-agent pipeline."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ss.agents.council.council import LLMCouncil
from ss.agents.jury import JuryAgent
from ss.agents.master import MasterAgent
from ss.agents.reception import ReceptionAgent
from ss.blackboard.models import Session, SessionStatus
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.intel.code_index import CodeIndex
from ss.loop.branch_loop import BranchLoop, BranchResult
from ss.memory.cleanup import MemoryCleanup
from ss.memory.client_profile import ClientProfileManager
from ss.memory.manager import MemoryManager
from ss.memory.project_memory import ProjectMemory
from ss.pipeline.events import EventType, create_event
from ss.skills.finder import SkillFinder
from ss.skills.resolver import SkillResolver

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionRunner:
    """Orchestrates the full SS pipeline for a given session.

    Manages session lifecycle: intake → strategies → parallel branches →
    jury scoring → synthesis → memory distribution.
    """

    def __init__(
        self,
        repo: Repository,
        vector_store: Any,
        memory_mgr: MemoryManager,
        bundle: Any,
        config: Config,
    ) -> None:
        self._repo = repo
        self._vector_store = vector_store
        self._memory_mgr = memory_mgr
        self._bundle = bundle
        self._config = config
        self._cleanup = MemoryCleanup(repo, config)
        self._profile_mgr = ClientProfileManager(repo)
        self._running: dict[str, asyncio.Task] = {}
        self._skill_resolver = SkillResolver(SkillFinder(config))

        self._pml = None
        if config.pml_enabled:
            try:
                self._pml = ProjectMemory(config.pml_dir)
            except Exception as exc:
                logger.warning("PML init failed — continuing without project memory: %s", exc)

        self._cil = None
        if config.cil_enabled:
            try:
                self._cil = CodeIndex(
                    config.cil_index_root,
                    config.cil_dir,
                    encoder=vector_store.encoder,
                    log_file=config.cil_log_file,
                )
            except Exception as exc:
                logger.warning("CIL init failed — continuing without code index: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def llm_available(self) -> bool:
        """Whether an LLM provider is available (delegates to the adapter bundle)."""
        return self._bundle.llm_available

    def bind_context(self, mcp_context: Any) -> None:
        """Attach a live MCP context to the bundle for the sampling fallback."""
        self._bundle.bind_context(mcp_context)

    async def solve(
        self,
        problem: str,
        context: dict[str, Any] | None = None,
        client_id: str = "default",
        wait: bool = True,
        timeout: float | None = -1.0,
    ) -> dict:
        """Start the pipeline; block until done (default) or return immediately.

        Args:
            problem: The problem text to solve.
            context: Optional context dict passed to agents.
            client_id: Client identifier for session grouping.
            wait: When False, return the session_id immediately without blocking.
            timeout: Controls the soft-timeout when ``wait=True``.

                - ``-1.0`` (default) — use ``config.solve_wait_timeout``.
                - ``None`` — await pipeline to *completion* (no timeout); the call
                  only returns once the pipeline finishes or raises.
                - positive float — use this value as the soft-timeout in seconds.

        With a finite timeout, if the soft timeout elapses, returns a poll-me
        response while the pipeline keeps running (shielded). With ``wait=False``,
        returns the session id at once.
        """
        # Resolve effective timeout
        if timeout == -1.0:
            effective_timeout: float | None = self._config.solve_wait_timeout
        else:
            effective_timeout = timeout  # None or explicit float

        session_id = await self.start(problem, context, client_id)
        if not wait:
            return {"session_id": session_id, "status": "started"}
        task = self._running.get(session_id)
        if task is None:
            return await self.get_result(session_id)

        try:
            if effective_timeout is None:
                # Run to completion — no soft timeout
                await task
            else:
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=effective_timeout
                )
        except asyncio.TimeoutError:
            return {
                "session_id": session_id,
                "status": "running",
                "message": (
                    f"Still running after {effective_timeout:.0f}s; "
                    f"call get_result with session_id '{session_id}' to retrieve the answer."
                ),
            }
        except Exception:
            # Pipeline raised; _run_pipeline already recorded SESSION_FAILED.
            return await self.get_result(session_id)
        return await self.get_result(session_id)

    async def start(
        self,
        problem: str,
        context: dict[str, Any] | None = None,
        client_id: str = "default",
    ) -> str:
        """Create a session and spawn the pipeline in the background.

        Returns:
            The new session_id immediately (pipeline runs async).
        """
        session = Session(
            client_id=client_id,
            problem_text=problem,
            problem_context=context,
        )
        self._repo.insert_session(session)

        event = create_event(
            session.id,
            # Use reception as the "system" agent for this bootstrap event
            __import__("ss.blackboard.models", fromlist=["AgentName"]).AgentName.reception,
            EventType.SESSION_CREATED,
            "init",
            {"session_id": session.id, "client_id": client_id},
        )
        self._repo.emit_event(event)

        task = asyncio.create_task(
            self._run_pipeline(session.id, problem, context, client_id)
        )
        self._running[session.id] = task

        return session.id

    async def get_events(self, session_id: str, after: int = 0) -> list[dict]:
        """Return persisted events for a session after the given event id."""
        events = self._repo.get_events(session_id, after=after if after > 0 else None)
        return [
            {
                "id": e.id,
                "event_type": e.event_type,
                "agent_name": str(e.agent_name),
                "phase": e.phase,
                "payload": e.payload,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in events
        ]

    async def get_result(self, session_id: str) -> dict:
        """Return the synthesized result for a completed session."""
        session = self._repo.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id} not found"}

        if session.status == SessionStatus.active:
            return {"status": "running", "session_id": session_id}

        if session.status == SessionStatus.failed:
            return {"status": "failed", "session_id": session_id}

        if session.status == SessionStatus.cancelled:
            return {"status": "cancelled", "session_id": session_id}

        # Completed — find master_synthesized event
        events = self._repo.get_events(session_id)
        synthesis_event = None
        for e in reversed(events):
            if e.event_type == str(EventType.MASTER_SYNTHESIZED):
                synthesis_event = e
                break

        result: dict[str, Any] = {
            "status": str(session.status),
            "session_id": session_id,
            "duration_ms": session.duration_ms,
        }
        if synthesis_event:
            result["synthesis"] = synthesis_event.payload
        return result

    async def cancel(self, session_id: str) -> dict:
        """Cancel an active session pipeline task."""
        task = self._running.get(session_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            self._running.pop(session_id, None)

        self._repo.update_session_status(
            session_id,
            SessionStatus.cancelled,
            completed_at=_now(),
        )
        return {"session_id": session_id, "status": "cancelled"}

    async def inspect(self, session_id: str) -> dict:
        """Return a full snapshot of session state."""
        session = self._repo.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id} not found"}

        issues = self._repo.get_issue_by_session(session_id)
        strategies = self._repo.list_strategies(session_id)
        missions = self._repo.list_missions_by_session(session_id)
        events = self._repo.get_events(session_id)
        notes = self._repo.list_agent_notes(session_id)

        return {
            "session": session.model_dump(mode="json"),
            "issues": [i.model_dump(mode="json") for i in issues],
            "strategies": [s.model_dump(mode="json") for s in strategies],
            "missions": [m.model_dump(mode="json") for m in missions],
            "event_count": len(events),
            "note_count": len(notes),
        }

    async def get_notes(
        self, session_id: str, agent_name: str | None = None
    ) -> list[dict]:
        """Return agent notes for a session, optionally filtered by agent."""
        from ss.blackboard.models import AgentName
        agent_enum = AgentName(agent_name) if agent_name else None
        notes = self._repo.list_agent_notes(session_id, agent_name=agent_enum)
        return [n.model_dump(mode="json") for n in notes]

    async def get_history(
        self, client_id: str | None = None, limit: int = 10
    ) -> list[dict]:
        """Return recent session history, optionally filtered by client."""
        sessions = self._repo.list_sessions(client_id=client_id)
        sessions = sessions[:limit]
        return [s.model_dump(mode="json") for s in sessions]

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        session_id: str,
        problem: str,
        context: dict[str, Any] | None,
        client_id: str,
    ) -> None:
        """Full pipeline: intake → strategies → branches → jury → synthesis."""
        start_time = _now()

        try:
            # Run memory cleanup at start of each session
            self._cleanup.cleanup_expired()

            # Ensure client profile exists
            self._profile_mgr.get_or_create(client_id)

            # ----------------------------------------------------------
            # Phase 1: Intake
            # ----------------------------------------------------------
            reception = ReceptionAgent(
                self._repo, self._bundle.for_agent("reception"), self._memory_mgr, self._vector_store,
                pml=self._pml,
            )
            issue = await reception.execute(session_id, problem=problem, context=context)

            master = MasterAgent(
                self._repo, self._bundle.for_agent("master"), self._memory_mgr, self._vector_store,
                pml=self._pml, cil=self._cil,
            )
            await master.execute(session_id, issue_summary=issue.summary, client_id=client_id)

            council = LLMCouncil(
                self._repo, self._bundle, self._memory_mgr, self._vector_store, self._config,
                pml=self._pml,
            )
            strategies = await council.execute(
                session_id, issue=issue, num_strategies=self._config.num_strategies
            )

            # ----------------------------------------------------------
            # Restrategize loop
            # ----------------------------------------------------------
            jury = JuryAgent(
                self._repo, self._bundle.for_agent("jury"), self._memory_mgr,
                self._vector_store, self._config,
            )

            succeeded_missions = []
            all_branch_results: list[BranchResult] = []
            failure_context: str | None = None

            for restrategize_round in range(self._config.max_restrategize_rounds + 1):
                if restrategize_round > 0:
                    strategies = await council.execute(
                        session_id, issue=issue, num_strategies=self._config.num_strategies,
                        failure_context=failure_context,
                    )

                branch = BranchLoop(
                    self._repo, self._bundle, self._memory_mgr, self._vector_store,
                    self._config, skill_resolver=self._skill_resolver,
                    pml=self._pml, cil=self._cil,
                )
                results = await asyncio.gather(
                    *[branch.run(session_id, s) for s in strategies],
                    return_exceptions=True,
                )

                all_branch_results = []
                succeeded_missions = []
                failed_reasons = []
                for r in results:
                    if isinstance(r, Exception):
                        failed_reasons.append(str(r))
                        continue
                    all_branch_results.append(r)
                    if r.status == "succeeded" and r.mission is not None:
                        succeeded_missions.append(r.mission)
                    else:
                        failed_reasons.append(r.reason)

                if succeeded_missions:
                    break
                failure_context = "; ".join(failed_reasons)
                if restrategize_round >= self._config.max_restrategize_rounds:
                    break

            # ----------------------------------------------------------
            # Phase 3: Evaluation
            # ----------------------------------------------------------
            scores = await jury.execute(session_id, missions=succeeded_missions)

            # Pick winner by highest weighted_total
            winning_strategy = None
            winning_score = 0.0
            winning_mission = None

            if scores:
                best = max(scores, key=lambda sc: sc.weighted_total)
                winning_score = best.weighted_total
                winning_strategy = self._repo.get_strategy(best.strategy_id)

                # Find the mission for the winning strategy
                for m in succeeded_missions:
                    if m.strategy_id == best.strategy_id:
                        winning_mission = m
                        break

            # Synthesize if we have a winner
            if winning_strategy is not None:
                mission_results = (
                    self._repo.list_mission_results(winning_mission.id)
                    if winning_mission
                    else []
                )
                await master.execute(
                    session_id,
                    winning_strategy=winning_strategy,
                    score=winning_score,
                    mission_results=mission_results,
                )

            # Distribute learnings for all strategies
            for score in scores:
                strategy = self._repo.get_strategy(score.strategy_id)
                if strategy is None:
                    continue
                mission_for_strategy = next(
                    (m for m in succeeded_missions if m.strategy_id == score.strategy_id),
                    None,
                )
                if mission_for_strategy is None:
                    continue
                await master.execute(
                    session_id,
                    mission=mission_for_strategy,
                    strategy=strategy,
                    score=score.weighted_total,
                )

            # Cross-session learning (permanent patterns/anti-patterns)
            winning_skills: list[str] = []
            if winning_strategy is not None:
                wr = next((r for r in all_branch_results
                           if r.mission is not None and r.strategy_id == winning_strategy.id), None)
                if wr is not None:
                    winning_skills = wr.required_skills
            failed_strategies = [
                self._repo.get_strategy(r.strategy_id)
                for r in all_branch_results if r.status == "failed"
            ]
            failed_strategies = [s for s in failed_strategies if s is not None]
            await master.extract_cross_session_learnings(
                session_id,
                winning_strategy=winning_strategy,
                failed_strategies=failed_strategies,
                winning_skills=winning_skills,
            )

            # Write project memory (PML decisions/timeline, CIL tasks)
            if self._pml is not None or self._cil is not None:
                await master.write_project_memory(
                    session_id,
                    winning_strategy=winning_strategy,
                    failed_strategies=failed_strategies,
                    why="",
                    timeline_entry="",
                    open_tasks=[],
                )

            # ----------------------------------------------------------
            # Mark session completed
            # ----------------------------------------------------------
            completed_at = _now()
            duration_ms = int((completed_at - start_time).total_seconds() * 1000)
            self._repo.update_session_status(
                session_id,
                SessionStatus.completed,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

            # Update client profile
            self._profile_mgr.update_after_session(client_id)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Emit SESSION_FAILED event
            from ss.blackboard.models import AgentName
            event = create_event(
                session_id,
                AgentName.master,
                EventType.SESSION_FAILED,
                "error",
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            self._repo.emit_event(event)

            self._repo.update_session_status(
                session_id,
                SessionStatus.failed,
                completed_at=_now(),
            )
            raise
        finally:
            self._running.pop(session_id, None)
