"""JudgeAgent: verify a taktik for correctness and feasibility."""
from __future__ import annotations

from typing import Any

from ss.agents.base import BaseAgent
from ss.blackboard.models import AgentName, MemoryType, NoteType
from ss.blackboard.repository import Repository
from ss.memory.project_memory import ProjectMemory
from ss.pipeline.events import EventType
from ss.sampling.adapter import SamplingAdapter


VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verified": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "issues_found": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verified", "reasoning", "issues_found", "suggestions"],
}


class JudgeAgent(BaseAgent):
    """Verifies a taktik plan against known failure patterns."""

    def __init__(
        self,
        repo: Repository,
        sampling: SamplingAdapter,
        memory_mgr: Any,
        vector_store: Any,
        pml: ProjectMemory | None = None,
    ) -> None:
        super().__init__(repo, sampling, memory_mgr, vector_store)
        self._pml = pml

    @property
    def name(self) -> AgentName:
        return AgentName.judge

    @property
    def persona(self) -> str:
        return (
            "You are the Judge Agent. Your role is to critically evaluate execution "
            "plans (taktiks) for correctness, completeness, and feasibility. You look "
            "for potential failure points, missing steps, and anti-patterns. Be rigorous "
            "and thorough in your assessment."
        )

    @property
    def temperature(self) -> float:
        return 0.1

    async def execute(
        self,
        session_id: str,
        *,
        taktik: Any,
        strategy: Any,
        issue: Any,
        **kwargs: Any,
    ) -> dict:
        """Verify a taktik plan.

        Args:
            session_id: The active session id.
            taktik: The Taktik object to verify.
            strategy: The Strategy this taktik implements.
            issue: The Issue being addressed.

        Returns:
            Dict with: verified, reasoning, issues_found, suggestions.
        """
        # Recall bad_practice memories to check for known failure patterns
        bad_practices = self.recall_memories(
            taktik.steps[0].instruction if taktik.steps else strategy.description,
            type=MemoryType.bad_practice,
            limit=5,
        )
        bad_ctx = ""
        if bad_practices:
            bad_ctx = "\n\nKnown failure patterns to check against:\n" + "\n".join(
                f"- {m.content}" for m in bad_practices
            )

        steps_text = "\n".join(
            f"  Step {s.index}: {s.instruction} → Expected: {s.expected_outcome}"
            for s in taktik.steps
        )

        system_prompt = (
            "Evaluate this execution plan (taktik) for correctness, completeness, and "
            "feasibility. Check for logical errors, missing steps, and known failure "
            "patterns. Return a structured JSON verdict."
        )
        user_message = (
            f"Issue: {issue.summary}\n"
            f"Strategy: {strategy.description}\n"
            f"Objective: {strategy.objective}\n\n"
            f"Taktik plan ({len(taktik.steps)} steps):\n{steps_text}\n"
            f"Required skills: {', '.join(taktik.required_skills)}\n"
            f"Estimated complexity: {taktik.estimated_complexity}"
        ) + bad_ctx

        # Append PML context if available
        if self._pml is not None:
            pml_context = self._pml.as_context(["DECISION"])
            if pml_context:
                user_message += (
                    f"\n\n{pml_context}\n\n"
                    "If the plan violates any project DECISION listed above, "
                    "set verified=false and name the violated decision in issues_found."
                )

        result = await self.llm_call_structured(
            system_prompt, user_message, VERIFICATION_SCHEMA
        )

        verified = bool(result.get("verified", False))
        reasoning = result.get("reasoning", "")
        issues_found = result.get("issues_found", [])
        suggestions = result.get("suggestions", [])

        # Update taktik verification in DB
        self._repo.update_taktik_verification(
            taktik.id,
            verified=verified,
            judge_verification={
                "reasoning": reasoning,
                "issues_found": issues_found,
                "suggestions": suggestions,
            },
        )

        # Emit event
        payload: dict = {
            "taktik_id": taktik.id,
            "verified": verified,
            "reasoning": reasoning,
        }
        if not verified:
            rejection_reason = "; ".join(issues_found) if issues_found else reasoning
            payload["rejection_reason"] = rejection_reason

        self.emit_event(session_id, EventType.JUDGE_VERIFIED, "verify", payload)

        # Write error analysis note if rejected
        if not verified:
            self.write_note(
                session_id,
                content=(
                    f"Taktik rejected: {reasoning}\n"
                    f"Issues: {', '.join(issues_found)}\n"
                    f"Suggestions: {', '.join(suggestions)}"
                ),
                note_type=NoteType.error_analysis,
                references=[{"type": "taktik", "id": taktik.id}],
            )

        return result
