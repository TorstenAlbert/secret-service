"""Council persona layer: named thinking lenses for the LLM Council."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CouncilPersona:
    """A named thinking lens applied to a council member."""
    key: str
    name: str
    lens: str
    temperature: float


RISK_ANALYST = CouncilPersona(
    "risk_analyst", "Risk Analyst",
    "Hunt failure modes; name the single biggest risk that the other voices will overlook.",
    temperature=0.5,
)
FIRST_PRINCIPLES = CouncilPersona(
    "first_principles", "First Principles",
    "Question the root assumptions; strip the problem back to fundamentals and rebuild the "
    "approach from scratch.",
    temperature=0.7,
)
AMBITION = CouncilPersona(
    "ambition", "Ambition",
    "Find the 10x approach — the most ambitious path that is still genuinely viable, not "
    "naive optimism.",
    temperature=0.9,
)
NAIVE_OUTSIDER = CouncilPersona(
    "naive_outsider", "Naive Outsider",
    "Assume zero domain context; question what insiders take for granted and surface the "
    "curse of knowledge.",
    temperature=0.8,
)
PRAGMATIST = CouncilPersona(
    "pragmatist", "Pragmatist",
    "Find the smallest executable first step; if a strategy has no concrete move that ships "
    "today, it is not yet an answer.",
    temperature=0.4,
)
CHAIRMAN = CouncilPersona(
    "chairman", "Chairman",
    "Neutral synthesis: distil the voices into ranked competing strategies with no persona "
    "bias — note agreement, genuine clashes, and the insight no single voice surfaced.",
    temperature=0.3,
)

DEFAULT_PERSONAS: list[CouncilPersona] = [
    RISK_ANALYST, FIRST_PRINCIPLES, AMBITION, NAIVE_OUTSIDER, PRAGMATIST
]
PERSONAS_BY_KEY: dict[str, CouncilPersona] = {
    p.key: p for p in (*DEFAULT_PERSONAS, CHAIRMAN)
}


def personas_for(keys: list[str]) -> list[CouncilPersona]:
    """Resolve persona keys to CouncilPersona objects, skipping unknown keys."""
    return [PERSONAS_BY_KEY[k] for k in keys if k in PERSONAS_BY_KEY]
