"""MemoryCleanup: expire and decay memories over time."""
from __future__ import annotations

from ss.blackboard.repository import Repository
from ss.config import Config


class MemoryCleanup:
    """Handles memory expiration and confidence decay."""

    def __init__(self, repo: Repository, config: Config) -> None:
        self._repo = repo
        self._config = config

    def cleanup_expired(self) -> int:
        """Deactivate memories whose expiry time has passed.

        Returns the number of memories deactivated.
        """
        return self._repo.deactivate_expired()

    def decay_confidence(self, memory_id: str, reason: str) -> None:
        """Reduce a memory's confidence by the configured decay rate.

        Clamps the result to a minimum of 0.0.
        """
        memory = self._repo.get_memory(memory_id)
        if memory is None:
            return
        new_confidence = max(0.0, memory.confidence - self._config.confidence_decay_rate)
        self._repo.update_memory_confidence(memory_id, new_confidence)
