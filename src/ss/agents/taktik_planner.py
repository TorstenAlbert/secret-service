"""TaktikPlannerAgent: create a detailed step-by-step plan for a strategy."""
from __future__ import annotations

import uuid
from typing import Any

from ss.agents.base import BaseAgent
from ss.blackboard.models import AgentName, MemoryType, Taktik, TaktikStep
from ss.pipeline.events import EventType
from ss.skills.resolver import ResolvedSkill


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

    def __init__(self, repo, sampling, memory_mgr, vector_store, skill_resolver=None, cil=None, pml=None) -> None:
        super().__init__(repo, sampling, memory_mgr, vector_store)
        self._skill_resolver = skill_resolver
        self._cil = cil
        self._pml = pml

    @staticmethod
    def _format_skills_context(resolved: list[ResolvedSkill]) -> str:
        if not resolved:
            return ""
        lines = "\n".join(
            f"- {r.skill.name} ({r.skill.source}): {r.skill.description}"
            for r in resolved
        )
        return (
            "\n\n## AVAILABLE SKILLS (discovered via the `npx skills` CLI)\n"
            "Use these named, installable skills in your plan where applicable "
            "(install with `npx skills add <source>`). Do NOT re-implement what "
            "these already provide.\n"
            f"{lines}\n"
        )

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

        # Step 0: find-skills pre-pass (best-effort)
        skills_ctx = ""
        if self._skill_resolver is not None:
            resolved_skills = await self._skill_resolver.resolve(strategy)
            skills_ctx = self._format_skills_context(resolved_skills)

        # Step 1: CIL grounding (code structure context)
        # CIL lowers token cost per call, not the number of calls (that is the loop budget's job).
        cil_ctx = ""
        if self._cil is not None:
            cil_summary = self._cil.summary_text()
            # Cap signatures so the planning prompt stays cheap even on a large
            # index (signatures("") spans the whole tree); CIL's value is fewer
            # tokens per call, so we include only a bounded structural sample.
            cil_sigs = self._cil.signatures("")[:20]
            cil_ctx = f"\n\n## CODE STRUCTURE (from index)\n{cil_summary}\n"
            if cil_sigs:
                cil_ctx += "Key signatures:\n" + "\n".join(
                    f"- {s.get('signature') or s.get('name')} ({s.get('file')}:{s.get('line')})"
                    for s in cil_sigs
                ) + "\n"

        # Step 2: PML decision constraints
        pml_ctx = ""
        if self._pml is not None:
            pml_decisions = self._pml.as_context(["DECISION"])
            if pml_decisions:
                pml_ctx = f"\n\n## PROJECT DECISIONS — your plan MUST NOT violate these\n{pml_decisions}"

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
        ) + cil_ctx + pml_ctx + skills_ctx + memory_ctx + rejection_ctx

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
