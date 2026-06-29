"""ProgressTracker + CheckResult for the per-branch agentic loop."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckResult:
    """Outcome of a postcondition check for one loop iteration."""
    passed: bool
    evidence: str
    next_action: str  # "continue" | "stop" | "escalate" | "budget_halt"


class ProgressTracker:
    """Tracks per-branch iterations and detects budget exhaustion / stagnation."""

    def __init__(self, max_iterations: int, no_progress_threshold: int) -> None:
        self._max = max_iterations
        self._threshold = no_progress_threshold
        self._results: list[CheckResult] = []

    @property
    def iterations(self) -> int:
        """Number of recorded iterations."""
        return len(self._results)

    def record(self, result: CheckResult) -> None:
        """Record one iteration's check result."""
        self._results.append(result)

    def budget_exhausted(self) -> bool:
        """True once the iteration budget is reached."""
        return len(self._results) >= self._max

    def is_stagnant(self, n: int | None = None) -> bool:
        """True if the last ``n`` results carry identical evidence (no progress)."""
        n = n or self._threshold
        if len(self._results) < n:
            return False
        recent = self._results[-n:]
        return len({r.evidence for r in recent}) == 1
