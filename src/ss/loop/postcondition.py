"""PostconditionChecker: decide whether a branch has met its acceptance criteria."""
from __future__ import annotations

from typing import Any

from ss.loop.progress import CheckResult

_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["YES", "NO"]},
        "explanation": {"type": "string"},
    },
    "required": ["verdict", "explanation"],
}


class PostconditionChecker:
    """Evaluates a postcondition against the branch's accumulated evidence.

    For now this is an LLM-based check. A low-temperature judge model is asked
    whether the postcondition holds given the evidence.
    """

    def __init__(self, adapter: Any, temperature: float = 0.1, cil: Any = None) -> None:
        self._adapter = adapter
        self._temperature = temperature
        self._cil = cil

    async def check(self, postcondition: str, session_context: dict) -> CheckResult:
        """Return whether the postcondition is satisfied given the context."""
        # TODO: replace with mechanical CLI/test runner in Phase 5.
        # When a CIL (CodeIndex log) is available, attempt a mechanical short-circuit
        # before consulting the LLM.
        cil_log_lines: list[dict] = []
        if self._cil is not None:
            errors = self._cil.log("query", level="error")
            if errors:
                evidence = "runtime errors: " + "; ".join(
                    e.get("message", "") for e in errors[:5]
                )
                return CheckResult(passed=False, evidence=evidence, next_action="continue")

            recent = self._cil.log("query", limit=20)
            for entry in recent:
                if entry.get("postcondition") == "met" or entry.get("level") == "success":
                    return CheckResult(
                        passed=True,
                        evidence="live-log success signal",
                        next_action="stop",
                    )
            # No mechanical verdict — fall through to LLM, but carry log lines as context.
            cil_log_lines = recent

        results = session_context.get("mission_results", [])
        evidence_lines = []
        for r in results:
            outcome = getattr(r, "actual_outcome", None) or (r.get("actual_outcome") if isinstance(r, dict) else str(r))
            evidence_lines.append(f"- {outcome}")
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "(no results yet)"

        system = (
            "You are a strict verification judge. Decide whether the postcondition is "
            "fully satisfied by the evidence. Answer YES only if clearly met."
        )
        user_parts = [
            f"Postcondition (acceptance criteria):\n{postcondition}\n\n"
            f"Evidence from mission results:\n{evidence_text}",
        ]
        if cil_log_lines:
            log_summary = "\n".join(
                f"  [{e.get('level','info')}] {e.get('message','')}" for e in cil_log_lines
            )
            user_parts.append(f"\nLive log (recent entries):\n{log_summary}")
        user_parts.append("\nReturn JSON with 'verdict' (YES/NO) and 'explanation'.")
        user = "\n".join(user_parts)

        data = await self._adapter.complete_structured(
            system, user, _CHECK_SCHEMA, temperature=self._temperature
        )
        passed = str(data.get("verdict", "NO")).strip().upper() == "YES"
        return CheckResult(
            passed=passed,
            evidence=data.get("explanation", ""),
            next_action="stop" if passed else "continue",
        )
