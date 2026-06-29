"""SkillResolver: turn a strategy into a set of resolved, reusable skills."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ss.skills.finder import FoundSkill, SkillFinder


@dataclass
class ResolvedSkill:
    """A capability mapped to the best matching skill from the registry."""
    capability: str
    skill: FoundSkill


class SkillResolver:
    """Resolves a strategy's capabilities to concrete skills via SkillFinder."""

    def __init__(self, finder: SkillFinder) -> None:
        self._finder = finder

    def _extract_capabilities(self, strategy: Any) -> list[str]:
        """Derive capability queries from the strategy. Kept deliberately simple."""
        description = getattr(strategy, "description", "")
        objective = getattr(strategy, "objective", "")
        query = description if not objective else f"{description} ({objective})"
        return [query] if query.strip() else []

    async def resolve(self, strategy: Any) -> list[ResolvedSkill]:
        """Resolve the strategy's capabilities to skills (best match per capability)."""
        resolved: list[ResolvedSkill] = []
        for capability in self._extract_capabilities(strategy):
            found = await self._finder.find(capability)
            if found:
                resolved.append(ResolvedSkill(capability=capability, skill=found[0]))
        return resolved
