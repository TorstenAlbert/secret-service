"""CouncilMember: one persona-driven voice that proposes and peer-reviews."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ss.agents.council.personas import CouncilPersona


@dataclass
class AnonymisedProposal:
    """A proposal stripped of its author's identity for blind peer review."""
    label: str
    content: str


def _anti_pattern_block(anti_patterns: list[str]) -> str:
    if not anti_patterns:
        return ""
    lines = "\n".join(f"- {a}" for a in anti_patterns)
    return (
        "\n\n## KNOWN ANTI-PATTERNS — DO NOT REPEAT\n"
        "These approaches have failed before. Penalise any proposal that repeats them.\n"
        f"{lines}\n"
    )


class CouncilMember:
    """A single council voice (persona) that can propose and review strategies."""

    def __init__(
        self, bundle: Any, persona: CouncilPersona, model: str, temperature: float | None = None
    ) -> None:
        self._bundle = bundle
        self._persona = persona
        self._model = model
        self._temperature = temperature if temperature is not None else persona.temperature

    @property
    def label(self) -> str:
        """The member's persona name (e.g. ``Risk Analyst``)."""
        return self._persona.name

    @property
    def persona(self) -> CouncilPersona:
        """The thinking-lens persona this member embodies."""
        return self._persona

    async def propose(
        self, issue: Any, anti_patterns: list[str], *, project_context: str = ""
    ) -> str:
        """Independently propose a strategy through this persona's lens."""
        system = (
            f"You are {self._persona.name}, one voice on an expert problem-solving council. "
            f"{self._persona.lens}\n\n"
            "Propose ONE distinct, viable strategy to solve the issue, seen through your lens. "
            "Be concrete and concise (4-6 sentences)."
        )
        user = (
            f"Issue summary: {getattr(issue, 'summary', issue)}\n"
            f"Classification: {getattr(issue, 'classification', 'unknown')}\n"
            f"Postcondition (acceptance criteria): {getattr(issue, 'postcondition', '')}"
            f"{_anti_pattern_block(anti_patterns)}"
            f"{project_context}"
        )
        return await self._bundle.council_complete(
            model=self._model,
            system_prompt=system,
            messages=[{"role": "user", "content": user}],
            temperature=self._temperature,
        )

    async def review(
        self,
        proposals: list[AnonymisedProposal],
        anti_patterns: list[str],
        *,
        project_context: str = "",
    ) -> str:
        """Critique anonymised peer proposals through this persona's lens."""
        system = (
            f"You are {self._persona.name}. {self._persona.lens}\n\n"
            "Review the anonymised peer proposals below. Through your lens, name the strongest "
            "proposal, the biggest blind spot, and — most importantly — what they ALL missed. "
            "Critique content, not identity."
        )
        proposal_text = "\n\n".join(f"### {p.label}\n{p.content}" for p in proposals)
        user = (
            f"{_anti_pattern_block(anti_patterns)}"
            f"\n## PROPOSALS TO REVIEW\n{proposal_text}"
            f"{project_context}"
        )
        return await self._bundle.council_complete(
            model=self._model,
            system_prompt=system,
            messages=[{"role": "user", "content": user}],
            # Peer review stays critical/low-temp regardless of the persona's
            # (higher) proposal temperature.
            temperature=0.3,
        )
