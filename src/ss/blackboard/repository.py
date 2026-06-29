"""Repository: CRUD operations for all SS blackboard entities."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from ss.blackboard.database import Database
from ss.blackboard.models import (
    AgentName,
    AgentNote,
    ClientProfile,
    IssueClassification,
    Issue,
    Memory,
    MemoryScope,
    MemoryType,
    Mission,
    MissionResult,
    NoteType,
    Session,
    SessionEvent,
    SessionStatus,
    StrategyScore,
    Strategy,
    Taktik,
    TaktikStep,
)


def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _dt_req(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    """Provides typed CRUD operations over the SS SQLite database."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._db.conn

    # -----------------------------------------------------------------------
    # Row mappers
    # -----------------------------------------------------------------------

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            client_id=row["client_id"],
            status=SessionStatus(row["status"]),
            problem_text=row["problem_text"],
            problem_context=json.loads(row["problem_context"]) if row["problem_context"] else None,
            created_at=_dt_req(row["created_at"]),
            updated_at=_dt_req(row["updated_at"]),
            completed_at=_dt(row["completed_at"]),
            duration_ms=row["duration_ms"],
            total_llm_calls=row["total_llm_calls"],
            total_events=row["total_events"],
        )

    def _row_to_issue(self, row: sqlite3.Row) -> Issue:
        return Issue(
            id=row["id"],
            session_id=row["session_id"],
            summary=row["summary"],
            classification=IssueClassification(row["classification"]),
            severity=row["severity"],
            who=row["who"],
            where_location=row["where_location"],
            why_reason=row["why_reason"],
            precondition=row["precondition"],
            postcondition=row["postcondition"],
            key_points=json.loads(row["key_points"]),
            tags=json.loads(row["tags"]),
            created_at=_dt_req(row["created_at"]),
        )

    def _row_to_strategy(self, row: sqlite3.Row) -> Strategy:
        return Strategy(
            id=row["id"],
            session_id=row["session_id"],
            description=row["description"],
            objective=row["objective"],
            approach_type=row["approach_type"],
            rank=row["rank"],
            confidence=row["confidence"],
            jury_score=row["jury_score"],
            jury_metrics=json.loads(row["jury_metrics"]) if row["jury_metrics"] else None,
            status=row["status"],
            rating_label=row["rating_label"],
            failure_reason=row["failure_reason"],
            created_at=_dt_req(row["created_at"]),
            completed_at=_dt(row["completed_at"]),
        )

    def _row_to_taktik(self, row: sqlite3.Row) -> Taktik:
        raw_steps = json.loads(row["steps"])
        steps = [TaktikStep(**s) for s in raw_steps]
        return Taktik(
            id=row["id"],
            strategy_id=row["strategy_id"],
            session_id=row["session_id"],
            steps=steps,
            required_skills=json.loads(row["required_skills"]),
            estimated_complexity=row["estimated_complexity"],
            judge_verification=json.loads(row["judge_verification"]) if row["judge_verification"] else None,
            verified=bool(row["verified"]),
            attempt_number=row["attempt_number"],
            created_at=_dt_req(row["created_at"]),
        )

    def _row_to_mission(self, row: sqlite3.Row) -> Mission:
        return Mission(
            id=row["id"],
            taktik_id=row["taktik_id"],
            strategy_id=row["strategy_id"],
            session_id=row["session_id"],
            status=row["status"],
            attempt_number=row["attempt_number"],
            started_at=_dt_req(row["started_at"]),
            completed_at=_dt(row["completed_at"]),
            duration_ms=row["duration_ms"],
        )

    def _row_to_mission_result(self, row: sqlite3.Row) -> MissionResult:
        return MissionResult(
            id=row["id"],
            mission_id=row["mission_id"],
            step_index=row["step_index"],
            action=row["action"],
            expected_outcome=row["expected_outcome"],
            actual_outcome=row["actual_outcome"],
            success=bool(row["success"]),
            error_detail=row["error_detail"],
            artifacts=json.loads(row["artifacts"]) if row["artifacts"] else None,
            duration_ms=row["duration_ms"],
            created_at=_dt_req(row["created_at"]),
        )

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            scope=MemoryScope(row["scope"]),
            source_session_id=row["source_session_id"],
            source_agent=AgentName(row["source_agent"]),
            content=row["content"],
            structured_content=json.loads(row["structured_content"]) if row["structured_content"] else None,
            tags=json.loads(row["tags"]),
            relevance_count=row["relevance_count"],
            last_recalled_at=_dt(row["last_recalled_at"]),
            confidence=row["confidence"],
            superseded_by=row["superseded_by"],
            created_at=_dt_req(row["created_at"]),
            expires_at=_dt(row["expires_at"]),
            is_active=bool(row["is_active"]),
        )

    def _row_to_agent_note(self, row: sqlite3.Row) -> AgentNote:
        return AgentNote(
            id=row["id"],
            session_id=row["session_id"],
            agent_name=AgentName(row["agent_name"]),
            note_type=NoteType(row["note_type"]),
            content=row["content"],
            note_references=json.loads(row["note_references"]),
            created_at=_dt_req(row["created_at"]),
        )

    def _row_to_client_profile(self, row: sqlite3.Row) -> ClientProfile:
        return ClientProfile(
            client_id=row["client_id"],
            display_name=row["display_name"],
            expertise_level=row["expertise_level"],
            known_domains=json.loads(row["known_domains"]),
            communication_style=row["communication_style"],
            preferences=json.loads(row["preferences"]),
            total_sessions=row["total_sessions"],
            first_seen_at=_dt_req(row["first_seen_at"]),
            last_seen_at=_dt_req(row["last_seen_at"]),
        )

    def _row_to_strategy_score(self, row: sqlite3.Row) -> StrategyScore:
        return StrategyScore(
            id=row["id"],
            strategy_id=row["strategy_id"],
            session_id=row["session_id"],
            correctness=row["correctness"],
            completeness=row["completeness"],
            elegance=row["elegance"],
            robustness=row["robustness"],
            efficiency=row["efficiency"],
            weighted_total=row["weighted_total"],
            reasoning=row["reasoning"],
            created_at=_dt_req(row["created_at"]),
        )

    def _row_to_session_event(self, row: sqlite3.Row) -> SessionEvent:
        return SessionEvent(
            id=row["id"],
            session_id=row["session_id"],
            agent_name=AgentName(row["agent_name"]),
            event_type=row["event_type"],
            phase=row["phase"],
            payload=json.loads(row["payload"]),
            timestamp=_dt_req(row["timestamp"]),
        )

    # -----------------------------------------------------------------------
    # Sessions
    # -----------------------------------------------------------------------

    def insert_session(self, session: Session) -> None:
        self._conn.execute(
            """
            INSERT INTO sessions
              (id, client_id, status, problem_text, problem_context,
               created_at, updated_at, completed_at, duration_ms,
               total_llm_calls, total_events)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.client_id,
                str(session.status),
                session.problem_text,
                json.dumps(session.problem_context) if session.problem_context is not None else None,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.completed_at.isoformat() if session.completed_at else None,
                session.duration_ms,
                session.total_llm_calls,
                session.total_events,
            ),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return self._row_to_session(row) if row else None

    def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        completed_at: datetime | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self._conn.execute(
            """
            UPDATE sessions
               SET status = ?, completed_at = ?, duration_ms = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                str(status),
                completed_at.isoformat() if completed_at else None,
                duration_ms,
                _now_iso(),
                session_id,
            ),
        )
        self._conn.commit()

    def increment_llm_calls(self, session_id: str, by: int = 1) -> None:
        self._conn.execute(
            "UPDATE sessions SET total_llm_calls = total_llm_calls + ?, updated_at = ? WHERE id = ?",
            (by, _now_iso(), session_id),
        )
        self._conn.commit()

    def list_sessions(
        self,
        client_id: str | None = None,
        status: SessionStatus | None = None,
    ) -> list[Session]:
        query = "SELECT * FROM sessions WHERE 1=1"
        params: list[Any] = []
        if client_id is not None:
            query += " AND client_id = ?"
            params.append(client_id)
        if status is not None:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_session(r) for r in rows]

    # -----------------------------------------------------------------------
    # Issues
    # -----------------------------------------------------------------------

    def insert_issue(self, issue: Issue) -> None:
        self._conn.execute(
            """
            INSERT INTO issues
              (id, session_id, summary, classification, severity,
               who, where_location, why_reason, precondition, postcondition,
               key_points, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                issue.id,
                issue.session_id,
                issue.summary,
                str(issue.classification),
                issue.severity,
                issue.who,
                issue.where_location,
                issue.why_reason,
                issue.precondition,
                issue.postcondition,
                json.dumps(issue.key_points),
                json.dumps(issue.tags),
                issue.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_issue_by_session(self, session_id: str) -> list[Issue]:
        rows = self._conn.execute(
            "SELECT * FROM issues WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [self._row_to_issue(r) for r in rows]

    # -----------------------------------------------------------------------
    # Strategies
    # -----------------------------------------------------------------------

    def insert_strategy(self, strategy: Strategy) -> None:
        self._conn.execute(
            """
            INSERT INTO strategies
              (id, session_id, description, objective, approach_type,
               rank, confidence, jury_score, jury_metrics,
               status, rating_label, failure_reason, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy.id,
                strategy.session_id,
                strategy.description,
                strategy.objective,
                strategy.approach_type,
                strategy.rank,
                strategy.confidence,
                strategy.jury_score,
                json.dumps(strategy.jury_metrics) if strategy.jury_metrics is not None else None,
                strategy.status,
                strategy.rating_label,
                strategy.failure_reason,
                strategy.created_at.isoformat(),
                strategy.completed_at.isoformat() if strategy.completed_at else None,
            ),
        )
        self._conn.commit()

    def get_strategy(self, strategy_id: str) -> Strategy | None:
        row = self._conn.execute(
            "SELECT * FROM strategies WHERE id = ?", (strategy_id,)
        ).fetchone()
        return self._row_to_strategy(row) if row else None

    def list_strategies(self, session_id: str) -> list[Strategy]:
        rows = self._conn.execute(
            "SELECT * FROM strategies WHERE session_id = ? ORDER BY rank",
            (session_id,),
        ).fetchall()
        return [self._row_to_strategy(r) for r in rows]

    def update_strategy_status(
        self,
        strategy_id: str,
        status: str,
        rating_label: str | None = None,
        failure_reason: str | None = None,
        jury_score: float | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        self._conn.execute(
            """
            UPDATE strategies
               SET status = ?, rating_label = ?, failure_reason = ?,
                   jury_score = ?, completed_at = ?
             WHERE id = ?
            """,
            (
                status,
                rating_label,
                failure_reason,
                jury_score,
                completed_at.isoformat() if completed_at else None,
                strategy_id,
            ),
        )
        self._conn.commit()

    def update_strategy_score(
        self, strategy_id: str, jury_score: float, jury_metrics: dict[str, Any] | None = None
    ) -> None:
        self._conn.execute(
            "UPDATE strategies SET jury_score = ?, jury_metrics = ? WHERE id = ?",
            (jury_score, json.dumps(jury_metrics) if jury_metrics is not None else None, strategy_id),
        )
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Taktiks
    # -----------------------------------------------------------------------

    def insert_taktik(self, taktik: Taktik) -> None:
        self._conn.execute(
            """
            INSERT INTO taktiks
              (id, strategy_id, session_id, steps, required_skills,
               estimated_complexity, judge_verification, verified,
               attempt_number, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                taktik.id,
                taktik.strategy_id,
                taktik.session_id,
                json.dumps([s.model_dump() for s in taktik.steps]),
                json.dumps(taktik.required_skills),
                taktik.estimated_complexity,
                json.dumps(taktik.judge_verification) if taktik.judge_verification is not None else None,
                int(taktik.verified),
                taktik.attempt_number,
                taktik.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def update_taktik_verification(
        self, taktik_id: str, verified: bool, judge_verification: dict[str, Any] | None = None
    ) -> None:
        self._conn.execute(
            "UPDATE taktiks SET verified = ?, judge_verification = ? WHERE id = ?",
            (int(verified), json.dumps(judge_verification) if judge_verification is not None else None, taktik_id),
        )
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Missions
    # -----------------------------------------------------------------------

    def insert_mission(self, mission: Mission) -> None:
        self._conn.execute(
            """
            INSERT INTO missions
              (id, taktik_id, strategy_id, session_id, status,
               attempt_number, started_at, completed_at, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mission.id,
                mission.taktik_id,
                mission.strategy_id,
                mission.session_id,
                mission.status,
                mission.attempt_number,
                mission.started_at.isoformat(),
                mission.completed_at.isoformat() if mission.completed_at else None,
                mission.duration_ms,
            ),
        )
        self._conn.commit()

    def update_mission_status(
        self,
        mission_id: str,
        status: str,
        completed_at: datetime | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE missions SET status = ?, completed_at = ?, duration_ms = ? WHERE id = ?",
            (
                status,
                completed_at.isoformat() if completed_at else None,
                duration_ms,
                mission_id,
            ),
        )
        self._conn.commit()

    def list_missions_by_session(self, session_id: str) -> list[Mission]:
        rows = self._conn.execute(
            "SELECT * FROM missions WHERE session_id = ? ORDER BY started_at",
            (session_id,),
        ).fetchall()
        return [self._row_to_mission(r) for r in rows]

    # -----------------------------------------------------------------------
    # Mission Results
    # -----------------------------------------------------------------------

    def insert_mission_result(self, result: MissionResult) -> None:
        self._conn.execute(
            """
            INSERT INTO mission_results
              (id, mission_id, step_index, action, expected_outcome,
               actual_outcome, success, error_detail, artifacts, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.id,
                result.mission_id,
                result.step_index,
                result.action,
                result.expected_outcome,
                result.actual_outcome,
                int(result.success),
                result.error_detail,
                json.dumps(result.artifacts) if result.artifacts is not None else None,
                result.duration_ms,
                result.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def list_mission_results(self, mission_id: str) -> list[MissionResult]:
        rows = self._conn.execute(
            "SELECT * FROM mission_results WHERE mission_id = ? ORDER BY step_index",
            (mission_id,),
        ).fetchall()
        return [self._row_to_mission_result(r) for r in rows]

    # -----------------------------------------------------------------------
    # Memories
    # -----------------------------------------------------------------------

    def insert_memory(self, memory: Memory) -> None:
        self._conn.execute(
            """
            INSERT INTO memories
              (id, type, scope, source_session_id, source_agent,
               content, structured_content, tags, relevance_count,
               last_recalled_at, confidence, superseded_by,
               created_at, expires_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                str(memory.type),
                str(memory.scope),
                memory.source_session_id,
                str(memory.source_agent),
                memory.content,
                json.dumps(memory.structured_content) if memory.structured_content is not None else None,
                json.dumps(memory.tags),
                memory.relevance_count,
                memory.last_recalled_at.isoformat() if memory.last_recalled_at else None,
                memory.confidence,
                memory.superseded_by,
                memory.created_at.isoformat(),
                memory.expires_at.isoformat() if memory.expires_at else None,
                int(memory.is_active),
            ),
        )
        self._conn.commit()

    def get_memory(self, memory_id: str) -> Memory | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def list_memories(
        self,
        type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        active_only: bool = True,
    ) -> list[Memory]:
        query = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []
        if type is not None:
            query += " AND type = ?"
            params.append(str(type))
        if scope is not None:
            query += " AND scope = ?"
            params.append(str(scope))
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def supersede_memory(self, memory_id: str, superseded_by: str) -> None:
        self._conn.execute(
            "UPDATE memories SET superseded_by = ?, is_active = 0 WHERE id = ?",
            (superseded_by, memory_id),
        )
        self._conn.commit()

    def update_memory_confidence(self, memory_id: str, confidence: float) -> None:
        self._conn.execute(
            "UPDATE memories SET confidence = ? WHERE id = ?",
            (confidence, memory_id),
        )
        self._conn.commit()

    def increment_recall(self, memory_id: str) -> None:
        self._conn.execute(
            "UPDATE memories SET relevance_count = relevance_count + 1, last_recalled_at = ? WHERE id = ?",
            (_now_iso(), memory_id),
        )
        self._conn.commit()

    def deactivate_expired(self) -> int:
        cursor = self._conn.execute(
            "UPDATE memories SET is_active = 0 WHERE expires_at IS NOT NULL AND expires_at <= ? AND is_active = 1",
            (_now_iso(),),
        )
        self._conn.commit()
        return cursor.rowcount

    # -----------------------------------------------------------------------
    # Agent Notes
    # -----------------------------------------------------------------------

    def insert_agent_note(self, note: AgentNote) -> None:
        self._conn.execute(
            """
            INSERT INTO agent_notes
              (id, session_id, agent_name, note_type, content, note_references, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note.id,
                note.session_id,
                str(note.agent_name),
                str(note.note_type),
                note.content,
                json.dumps(note.note_references),
                note.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def list_agent_notes(
        self, session_id: str, agent_name: AgentName | None = None
    ) -> list[AgentNote]:
        query = "SELECT * FROM agent_notes WHERE session_id = ?"
        params: list[Any] = [session_id]
        if agent_name is not None:
            query += " AND agent_name = ?"
            params.append(str(agent_name))
        query += " ORDER BY created_at"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_agent_note(r) for r in rows]

    # -----------------------------------------------------------------------
    # Session Events
    # -----------------------------------------------------------------------

    def emit_event(self, event: SessionEvent) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO session_events
              (session_id, agent_name, event_type, phase, payload, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.session_id,
                str(event.agent_name),
                event.event_type,
                event.phase,
                json.dumps(event.payload),
                event.timestamp.isoformat(),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_events(
        self, session_id: str, after: int | None = None
    ) -> list[SessionEvent]:
        query = "SELECT * FROM session_events WHERE session_id = ?"
        params: list[Any] = [session_id]
        if after is not None:
            query += " AND id > ?"
            params.append(after)
        query += " ORDER BY id"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_session_event(r) for r in rows]

    # -----------------------------------------------------------------------
    # Strategy Scores
    # -----------------------------------------------------------------------

    def insert_strategy_score(self, score: StrategyScore) -> None:
        self._conn.execute(
            """
            INSERT INTO strategy_scores
              (id, strategy_id, session_id, correctness, completeness,
               elegance, robustness, efficiency, weighted_total, reasoning, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                score.id,
                score.strategy_id,
                score.session_id,
                score.correctness,
                score.completeness,
                score.elegance,
                score.robustness,
                score.efficiency,
                score.weighted_total,
                score.reasoning,
                score.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Client Profiles
    # -----------------------------------------------------------------------

    def upsert_client_profile(self, profile: ClientProfile) -> None:
        self._conn.execute(
            """
            INSERT INTO client_profiles
              (client_id, display_name, expertise_level, known_domains,
               communication_style, preferences, total_sessions,
               first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_id) DO UPDATE SET
              display_name = excluded.display_name,
              expertise_level = excluded.expertise_level,
              known_domains = excluded.known_domains,
              communication_style = excluded.communication_style,
              preferences = excluded.preferences,
              total_sessions = excluded.total_sessions,
              last_seen_at = excluded.last_seen_at
            """,
            (
                profile.client_id,
                profile.display_name,
                profile.expertise_level,
                json.dumps(profile.known_domains),
                profile.communication_style,
                json.dumps(profile.preferences),
                profile.total_sessions,
                profile.first_seen_at.isoformat(),
                profile.last_seen_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_client_profile(self, client_id: str) -> ClientProfile | None:
        row = self._conn.execute(
            "SELECT * FROM client_profiles WHERE client_id = ?", (client_id,)
        ).fetchone()
        return self._row_to_client_profile(row) if row else None
