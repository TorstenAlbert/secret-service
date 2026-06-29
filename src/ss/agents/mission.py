"""MissionAgent: execute a taktik step by step and record results."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ss.agents.base import BaseAgent
from ss.blackboard.models import AgentName, Mission, MissionResult
from ss.blackboard.repository import Repository
from ss.pipeline.events import EventType
from ss.sampling.adapter import SamplingAdapter


STEP_EXECUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "actual_outcome": {"type": "string"},
        "success": {"type": "boolean"},
        "error_detail": {"type": "string"},
    },
    "required": ["action", "actual_outcome", "success"],
}


class MissionAgent(BaseAgent):
    """Executes a taktik by simulating each step via LLM."""

    def __init__(
        self,
        repo: Repository,
        sampling: SamplingAdapter,
        memory_mgr: Any,
        vector_store: Any,
        cil: Any = None,
    ) -> None:
        """Initialize the MissionAgent.

        Args:
            repo: Repository for blackboard access.
            sampling: Sampling adapter for LLM calls.
            memory_mgr: Memory manager for recall.
            vector_store: Vector store for embeddings.
            cil: Optional Code Index Layer for precise locations and session notes.
                 CIL lowers token cost per call, not the number of calls.
        """
        super().__init__(repo, sampling, memory_mgr, vector_store)
        self._cil = cil

    @property
    def name(self) -> AgentName:
        return AgentName.mission

    @property
    def persona(self) -> str:
        return (
            "You are the Mission Agent. Your role is to execute software engineering "
            "tasks step by step. For each step, you perform the action, observe the "
            "outcome, and report whether it succeeded. Be precise and realistic about "
            "what each step actually accomplishes."
        )

    @property
    def temperature(self) -> float:
        return 0.2

    async def execute(
        self,
        session_id: str,
        *,
        taktik: Any,
        strategy_id: str,
        **kwargs: Any,
    ) -> Mission:
        """Execute a taktik step by step.

        Args:
            session_id: The active session id.
            taktik: The Taktik object with steps to execute.
            strategy_id: ID of the strategy this mission belongs to.

        Returns:
            The created and updated Mission object.
        """
        mission = Mission(
            id=str(uuid.uuid4()),
            taktik_id=taktik.id,
            strategy_id=strategy_id,
            session_id=session_id,
            status="running",
            attempt_number=taktik.attempt_number,
        )
        self._repo.insert_mission(mission)

        self.emit_event(
            session_id,
            EventType.MISSION_STARTED,
            "start",
            {
                "mission_id": mission.id,
                "taktik_id": taktik.id,
                "strategy_id": strategy_id,
                "total_steps": len(taktik.steps),
            },
        )

        succeeded = True
        start_time = datetime.now(timezone.utc)

        for step in taktik.steps:
            system_prompt = (
                "Execute this software engineering step and report what action was "
                "taken, what the actual outcome was, and whether it succeeded."
            )
            user_message = (
                f"Step {step.index}: {step.instruction}\n"
                f"Expected outcome: {step.expected_outcome}"
            )

            # Before LLM call: if CIL is set, query for relevant code locations
            if self._cil is not None:
                # Derive a key term from the step instruction
                # (first word longer than 3 chars, or first 40 chars)
                words = [w for w in step.instruction.split() if len(w) > 3]
                key_term = words[0] if words else step.instruction[:40]

                # Query CIL for relevant locations
                locations = self._cil.query(key_term, mode="contains", limit=5)

                # Append locations to the user message if we got hits
                if locations:
                    locations_block = "Relevant code locations:\n"
                    for loc in locations:
                        locations_block += f"  {loc['name']} ({loc['file']}:{loc['line']})\n"
                    user_message += f"\n{locations_block}"

            step_data = await self.llm_call_structured(
                system_prompt, user_message, STEP_EXECUTION_SCHEMA
            )

            step_success = bool(step_data.get("success", False))
            result = MissionResult(
                id=str(uuid.uuid4()),
                mission_id=mission.id,
                step_index=step.index,
                action=step_data.get("action", step.instruction),
                expected_outcome=step.expected_outcome,
                actual_outcome=step_data.get("actual_outcome", ""),
                success=step_success,
                error_detail=step_data.get("error_detail"),
            )
            self._repo.insert_mission_result(result)

            # After recording the step result: if CIL is set, persist the outcome as a note
            if self._cil is not None:
                area = f"mission:{mission.id}"
                note_text = f"step {step.index}: {result.actual_outcome[:200]}"
                self._cil.note(area, note_text)

            self.emit_event(
                session_id,
                EventType.MISSION_STEP,
                f"step_{step.index}",
                {
                    "mission_id": mission.id,
                    "step_index": step.index,
                    "success": step_success,
                    "action": result.action,
                    "actual_outcome": result.actual_outcome,
                },
            )

            if not step_success:
                succeeded = False
                break  # Stop on first failure

        # Update mission status
        completed_at = datetime.now(timezone.utc)
        duration_ms = int((completed_at - start_time).total_seconds() * 1000)
        final_status = "succeeded" if succeeded else "failed"
        self._repo.update_mission_status(
            mission.id,
            status=final_status,
            completed_at=completed_at,
            duration_ms=duration_ms,
        )

        self.emit_event(
            session_id,
            EventType.MISSION_COMPLETED,
            "complete",
            {
                "mission_id": mission.id,
                "status": final_status,
                "duration_ms": duration_ms,
            },
        )

        # Return updated mission object
        mission.status = final_status
        mission.completed_at = completed_at
        mission.duration_ms = duration_ms
        return mission
