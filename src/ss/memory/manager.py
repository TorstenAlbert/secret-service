"""MemoryManager: store and recall memories with vector similarity."""
from __future__ import annotations

from typing import Any

from ss.blackboard.models import Memory, MemoryScope, MemoryType
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.vectors.store import VectorStore


class MemoryManager:
    """Manages storing and recalling agent memories.

    Uses vector similarity to detect near-duplicates on store, and filters
    recalled memories by type, scope, confidence, and active status.
    """

    def __init__(self, repo: Repository, vector_store: VectorStore, config: Config) -> None:
        self._repo = repo
        self._vector_store = vector_store
        self._config = config

    def recall(
        self,
        query: str,
        type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 5,
        min_confidence: float = 0.3,
    ) -> list[Memory]:
        """Search for relevant memories by vector similarity.

        Fetches candidates from the vector store, retrieves full Memory objects,
        filters by type/scope/confidence/is_active, increments recall counts,
        and returns up to ``limit`` results.
        """
        # Get more candidates than needed to allow for filtering
        candidates = self._vector_store.search("memory", query, limit=limit * 4)

        results: list[Memory] = []
        for hit in candidates:
            memory = self._repo.get_memory(hit["entity_id"])
            if memory is None:
                continue
            if not memory.is_active:
                continue
            if memory.confidence < min_confidence:
                continue
            if type is not None and memory.type != type:
                continue
            if scope is not None and memory.scope != scope:
                continue
            self._repo.increment_recall(memory.id)
            results.append(memory)
            if len(results) >= limit:
                break

        return results

    def store(self, memory: Memory) -> str:
        """Persist a memory, superseding near-duplicates first.

        Searches for existing memories with similar content (distance below
        ``config.similarity_threshold``) and supersedes them before inserting
        and indexing the new memory.

        Returns the id of the stored memory.
        """
        # Look for near-duplicates
        similar = self._vector_store.search("memory", memory.content, limit=5)
        for hit in similar:
            if hit["distance"] < self._config.similarity_threshold:
                existing = self._repo.get_memory(hit["entity_id"])
                if existing is not None and existing.is_active:
                    self._repo.supersede_memory(existing.id, memory.id)

        # Insert and index the new memory
        self._repo.insert_memory(memory)
        self._vector_store.index("memory", memory.id, memory.content)

        return memory.id
