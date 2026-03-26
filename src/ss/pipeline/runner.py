"""SessionRunner: orchestrates the full multi-agent pipeline."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from ss.agents.jury import JuryAgent
from ss.agents.master import MasterAgent
from ss.agents.reception import ReceptionAgent
from ss.agents.strategist import StrategistAgent
from ss.blackboard.models import Session, SessionStatus
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.memory.cleanup import MemoryCleanup
from ss.memory.client_profile import ClientProfileManager
from ss.memory.manager import MemoryManager
from ss.pipeline.branch import StrategyBranch, StrategyExhausted
from ss.pipeline.events import EventType, create_event


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
        sampling: Any,
        config: Config,
    ) -> None:
        self._repo = repo
        self._vector_store = vector_store
        self._memory_mgr = memory_mgr
        self._sampling = sampling
        self._config = config
        self._cleanup = MemoryCleanup(repo, config)
        self._profile_mgr = ClientProfileManager(repo)
        self._running: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
                self._repo, self._sampling, self._memory_mgr, self._vector_store
            )
            issue = await reception.execute(
                session_id, problem=problem, context=context
            )

            master = MasterAgent(
                self._repo, self._sampling, self._memory_mgr, self._vector_store
            )
            await master.execute(
                session_id,
                issue_summary=issue.summary,
                client_id=client_id,
            )

            strategist = StrategistAgent(
                self._repo, self._sampling, self._memory_mgr, self._vector_store
            )

            strategies = await strategist.execute(
                session_id,
                issue=issue,
                num_strategies=self._config.num_strategies,
            )

            # ----------------------------------------------------------
            # Restrategize loop
            # ----------------------------------------------------------
            jury = JuryAgent(
                self._repo, self._sampling, self._memory_mgr, self._vector_store,
                self._config,
            )

            succeeded_missions = []
            failure_context: str | None = None

            for restrategize_round in range(self._config.max_restrategize_rounds + 1):
                if restrategize_round > 0:
                    # All branches failed — ask strategist to re-evaluate
                    strategies = await strategist.execute(
                        session_id,
                        issue=issue,
                        num_strategies=self._config.num_strategies,
                        failure_context=failure_context,
                    )

                # --------------------------------------------------
                # Phase 2: Parallel strategy branches
                # --------------------------------------------------
                branch = StrategyBranch(
                    self._repo,
                    self._sampling,
                    self._memory_mgr,
                    self._vector_store,
                    self._config,
                )

                branch_results = await asyncio.gather(
                    *[branch.run(session_id, s) for s in strategies],
                    return_exceptions=True,
                )

                succeeded_missions = []
                failed_reasons = []
                for result in branch_results:
                    if isinstance(result, Exception):
                        reason = str(result)
                        failed_reasons.append(reason)
                    else:
                        succeeded_missions.append(result)

                if succeeded_missions:
                    break  # At least one branch succeeded

                # Build failure context for re-strategizing
                failure_context = "; ".join(failed_reasons)

                if restrategize_round >= self._config.max_restrategize_rounds:
                    # All rounds exhausted
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
