"""LLM Council: persona-driven multi-model strategy generation with peer review."""
from ss.agents.council.chairman import Chairman
from ss.agents.council.council import LLMCouncil
from ss.agents.council.member import AnonymisedProposal, CouncilMember
from ss.agents.council.personas import (
    CHAIRMAN, DEFAULT_PERSONAS, PERSONAS_BY_KEY, CouncilPersona, personas_for,
)

__all__ = [
    "AnonymisedProposal", "CouncilMember", "Chairman", "LLMCouncil",
    "CouncilPersona", "DEFAULT_PERSONAS", "PERSONAS_BY_KEY", "CHAIRMAN", "personas_for",
]
