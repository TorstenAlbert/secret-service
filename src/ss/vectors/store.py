"""Vector store: maps entity types to vss tables and manages embeddings."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from ss.blackboard.database import Database
from ss.vectors.encoder import EmbeddingEncoder


VSS_TABLES: dict[str, str] = {
    "issue": "vss_issues",
    "strategy": "vss_strategies",
    "taktik": "vss_taktiks",
    "mission": "vss_missions",
    "memory": "vss_memories",
    "agent_note": "vss_agent_notes",
}


def _content_hash(text: str) -> str:
    """Return SHA-256 hex digest of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VectorStore:
    """Manages embedding indexing and vector similarity search."""

    def __init__(self, db: Database, encoder: EmbeddingEncoder) -> None:
        self._db = db
        self._encoder = encoder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def encoder(self) -> EmbeddingEncoder:
        """The embedding encoder backing this store."""
        return self._encoder

    def index(self, entity_type: str, entity_id: str, text: str) -> None:
        """Embed text and insert/update in vss table + embedding_registry.

        Uses content_hash to skip re-embedding when content has not changed.
        """
        vss_table = self._vss_table(entity_type)
        content_hash = _content_hash(text)
        conn = self._db.conn

        # Check if already indexed with the same content hash
        row = conn.execute(
            "SELECT rowid, content_hash FROM embedding_registry WHERE entity_type = ? AND entity_id = ?",
            (entity_type, entity_id),
        ).fetchone()

        if row is not None and row["content_hash"] == content_hash:
            # Content unchanged — skip re-embedding
            return

        embedding_bytes = self._encoder.encode_bytes(text)

        if row is None:
            # New entity: insert into registry to get rowid, then insert into vss
            cursor = conn.execute(
                """
                INSERT INTO embedding_registry (entity_type, entity_id, content_hash, embedded_at)
                VALUES (?, ?, ?, ?)
                """,
                (entity_type, entity_id, content_hash, _now_iso()),
            )
            rowid = cursor.lastrowid
            conn.execute(
                f"INSERT INTO {vss_table} (rowid, embedding) VALUES (?, ?)",
                (rowid, embedding_bytes),
            )
        else:
            # Existing entity with changed content: update both tables
            rowid = row["rowid"]
            conn.execute(
                """
                UPDATE embedding_registry
                   SET content_hash = ?, embedded_at = ?
                 WHERE entity_type = ? AND entity_id = ?
                """,
                (content_hash, _now_iso(), entity_type, entity_id),
            )
            # sqlite-vss supports deletion + re-insertion for updates
            conn.execute(
                f"DELETE FROM {vss_table} WHERE rowid = ?",
                (rowid,),
            )
            conn.execute(
                f"INSERT INTO {vss_table} (rowid, embedding) VALUES (?, ?)",
                (rowid, embedding_bytes),
            )

        conn.commit()

    def search(
        self, entity_type: str, query: str, limit: int = 10
    ) -> list[dict]:
        """Search for similar entities.

        Returns a list of dicts with keys ``entity_id`` and ``distance``.
        Returns an empty list if no embeddings are indexed for this entity type.
        """
        vss_table = self._vss_table(entity_type)
        conn = self._db.conn

        # Check if any embeddings exist for this entity type
        count = conn.execute(
            "SELECT COUNT(*) FROM embedding_registry WHERE entity_type = ?",
            (entity_type,),
        ).fetchone()[0]
        if count == 0:
            return []

        query_bytes = self._encoder.encode_bytes(query)
        rows = conn.execute(
            f"""
            SELECT v.rowid, v.distance
              FROM {vss_table} v
             WHERE vss_search(v.embedding, ?)
             LIMIT ?
            """,
            (query_bytes, limit),
        ).fetchall()

        results = []
        for row in rows:
            reg_row = conn.execute(
                "SELECT entity_id FROM embedding_registry WHERE rowid = ?",
                (row["rowid"],),
            ).fetchone()
            if reg_row is not None:
                results.append(
                    {"entity_id": reg_row["entity_id"], "distance": float(row["distance"])}
                )
        return results

    def find_similar(
        self, entity_type: str, entity_id: str, limit: int = 5
    ) -> list[dict]:
        """Find entities similar to an existing one, excluding itself.

        Returns a list of dicts with keys ``entity_id`` and ``distance``.
        Returns an empty list if the entity is not indexed or no others exist.
        """
        vss_table = self._vss_table(entity_type)
        conn = self._db.conn

        # Fetch the entity's embedding via its rowid in the registry
        reg_row = conn.execute(
            "SELECT rowid FROM embedding_registry WHERE entity_type = ? AND entity_id = ?",
            (entity_type, entity_id),
        ).fetchone()
        if reg_row is None:
            return []

        rowid = reg_row["rowid"]

        # Retrieve the stored embedding bytes
        emb_row = conn.execute(
            f"SELECT embedding FROM {vss_table} WHERE rowid = ?",
            (rowid,),
        ).fetchone()
        if emb_row is None:
            return []

        embedding_bytes = emb_row["embedding"]

        # Search, requesting limit+1 to account for self
        rows = conn.execute(
            f"""
            SELECT v.rowid, v.distance
              FROM {vss_table} v
             WHERE vss_search(v.embedding, ?)
             LIMIT ?
            """,
            (embedding_bytes, limit + 1),
        ).fetchall()

        results = []
        for row in rows:
            if row["rowid"] == rowid:
                continue  # skip self
            reg = conn.execute(
                "SELECT entity_id FROM embedding_registry WHERE rowid = ?",
                (row["rowid"],),
            ).fetchone()
            if reg is not None:
                results.append(
                    {"entity_id": reg["entity_id"], "distance": float(row["distance"])}
                )
            if len(results) >= limit:
                break

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vss_table(self, entity_type: str) -> str:
        table = VSS_TABLES.get(entity_type)
        if table is None:
            raise ValueError(
                f"Unknown entity type: {entity_type!r}. "
                f"Valid types: {list(VSS_TABLES)}"
            )
        return table
