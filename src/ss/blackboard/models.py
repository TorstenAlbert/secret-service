"""Pydantic models for the SS blackboard (shared data layer)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SessionStatus(StrEnum):
    active = "active"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class IssueClassification(StrEnum):
    bug = "bug"
    architecture = "architecture"
    performance = "performance"
    refactor = "refactor"
    security = "security"
    testing = "testing"
    deployment = "deployment"
    documentation = "documentation"
    unknown = "unknown"


class AgentName(StrEnum):
    reception = "reception"
    master = "master"
    strategist = "strategist"
    taktik_planner = "taktik_planner"
    judge = "judge"
    mission = "mission"
    jury = "jury"


class MemoryType(StrEnum):
    good_practice = "good_practice"
    bad_practice = "bad_practice"
    client_identity = "client_identity"
    knowledge = "knowledge"
    pattern = "pattern"
    anti_pattern = "anti_pattern"
    insight = "insight"


class MemoryScope(StrEnum):
    short_term = "short_term"
    long_term = "long_term"
    permanent = "permanent"


class NoteType(StrEnum):
    observation = "observation"
    concern = "concern"
    decision = "decision"
    discovery = "discovery"
    contradiction = "contradiction"
    recommendation = "recommendation"
    error_analysis = "error_analysis"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Session(BaseModel):
    id: str = Field(default_factory=_uuid)
    client_id: str
    status: SessionStatus = SessionStatus.active
    problem_text: str
    problem_context: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None
    duration_ms: int | None = None
    total_llm_calls: int = 0
    total_events: int = 0


class Issue(BaseModel):
    id: str = Field(default_factory=_uuid)
    session_id: str
    summary: str
    classification: IssueClassification
    severity: str = "medium"
    who: str
    where_location: str
    why_reason: str
    precondition: str
    postcondition: str
    key_points: list[str]
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class Strategy(BaseModel):
    id: str = Field(default_factory=_uuid)
    session_id: str
    description: str
    objective: str
    approach_type: str | None = None
    rank: int
    confidence: float | None = None
    jury_score: float | None = None
    jury_metrics: dict[str, Any] | None = None
    status: str = "planned"
    rating_label: str | None = None
    failure_reason: str | None = None
    created_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None


class TaktikStep(BaseModel):
    index: int
    instruction: str
    expected_outcome: str


class Taktik(BaseModel):
    id: str = Field(default_factory=_uuid)
    strategy_id: str
    session_id: str
    steps: list[TaktikStep]
    required_skills: list[str] = Field(default_factory=list)
    estimated_complexity: str | None = None
    judge_verification: dict[str, Any] | None = None
    verified: bool = False
    attempt_number: int = 1
    created_at: datetime = Field(default_factory=_now)


class Mission(BaseModel):
    id: str = Field(default_factory=_uuid)
    taktik_id: str
    strategy_id: str
    session_id: str
    status: str = "running"
    attempt_number: int = 1
    started_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None
    duration_ms: int | None = None


class MissionResult(BaseModel):
    id: str = Field(default_factory=_uuid)
    mission_id: str
    step_index: int
    action: str
    expected_outcome: str | None = None
    actual_outcome: str
    success: bool
    error_detail: str | None = None
    artifacts: dict[str, Any] | None = None
    duration_ms: int | None = None
    created_at: datetime = Field(default_factory=_now)


class Memory(BaseModel):
    id: str = Field(default_factory=_uuid)
    type: MemoryType
    scope: MemoryScope
    source_session_id: str | None = None
    source_agent: AgentName
    content: str
    structured_content: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    relevance_count: int = 0
    last_recalled_at: datetime | None = None
    confidence: float = 1.0
    superseded_by: str | None = None
    created_at: datetime = Field(default_factory=_now)
    expires_at: datetime | None = None
    is_active: bool = True


class AgentNote(BaseModel):
    id: str = Field(default_factory=_uuid)
    session_id: str
    agent_name: AgentName
    note_type: NoteType = NoteType.observation
    content: str
    note_references: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class ClientProfile(BaseModel):
    client_id: str
    display_name: str | None = None
    expertise_level: str | None = None
    known_domains: list[str] = Field(default_factory=list)
    communication_style: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)
    total_sessions: int = 0
    first_seen_at: datetime = Field(default_factory=_now)
    last_seen_at: datetime = Field(default_factory=_now)


class ClientIssueHistory(BaseModel):
    id: str = Field(default_factory=_uuid)
    client_id: str
    session_id: str
    issue_summary: str
    classification: IssueClassification
    outcome: str
    winning_strategy_summary: str | None = None
    jury_score: float | None = None
    created_at: datetime = Field(default_factory=_now)


class StrategyScore(BaseModel):
    id: str = Field(default_factory=_uuid)
    strategy_id: str
    session_id: str
    correctness: float
    completeness: float
    elegance: float
    robustness: float
    efficiency: float
    weighted_total: float
    reasoning: str
    created_at: datetime = Field(default_factory=_now)


class SessionEvent(BaseModel):
    id: int | None = None
    session_id: str
    agent_name: AgentName
    event_type: str
    phase: str | None = None
    payload: dict[str, Any]
    timestamp: datetime = Field(default_factory=_now)
