"""Tests for Pydantic models."""
from datetime import datetime, timezone

import pytest

from ss.blackboard.models import (
    AgentName,
    AgentNote,
    ClientIssueHistory,
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


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

def test_session_status_values():
    assert SessionStatus.active == "active"
    assert SessionStatus.completed == "completed"
    assert SessionStatus.failed == "failed"
    assert SessionStatus.cancelled == "cancelled"


def test_issue_classification_values():
    assert IssueClassification.bug == "bug"
    assert IssueClassification.unknown == "unknown"


def test_agent_name_values():
    assert AgentName.reception == "reception"
    assert AgentName.master == "master"
    assert AgentName.jury == "jury"


def test_memory_type_values():
    assert MemoryType.good_practice == "good_practice"
    assert MemoryType.anti_pattern == "anti_pattern"


def test_memory_scope_values():
    assert MemoryScope.short_term == "short_term"
    assert MemoryScope.permanent == "permanent"


def test_note_type_values():
    assert NoteType.observation == "observation"
    assert NoteType.error_analysis == "error_analysis"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def test_session_defaults():
    s = Session(client_id="c1", problem_text="bug in prod")
    assert s.id  # auto-generated uuid
    assert s.status == SessionStatus.active
    assert s.problem_context is None
    assert s.completed_at is None
    assert s.duration_ms is None
    assert s.total_llm_calls == 0
    assert s.total_events == 0
    assert isinstance(s.created_at, datetime)
    assert isinstance(s.updated_at, datetime)


def test_session_custom():
    s = Session(client_id="c1", problem_text="test", status=SessionStatus.completed, total_llm_calls=5)
    assert s.status == SessionStatus.completed
    assert s.total_llm_calls == 5


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------

def test_issue_defaults():
    issue = Issue(
        session_id="s1",
        summary="NPE in handler",
        classification=IssueClassification.bug,
        who="backend team",
        where_location="handler.py:42",
        why_reason="null check missing",
        precondition="request with null field",
        postcondition="500 error returned",
        key_points=["null check", "input validation"],
    )
    assert issue.id
    assert issue.severity == "medium"
    assert issue.tags == []
    assert isinstance(issue.created_at, datetime)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def test_strategy_defaults():
    s = Strategy(
        session_id="s1",
        description="Add null check",
        objective="Prevent NPE",
        rank=1,
    )
    assert s.id
    assert s.status == "planned"
    assert s.confidence is None
    assert s.jury_score is None
    assert s.jury_metrics is None
    assert s.approach_type is None
    assert s.completed_at is None


# ---------------------------------------------------------------------------
# TaktikStep and Taktik
# ---------------------------------------------------------------------------

def test_taktik_step():
    step = TaktikStep(index=0, instruction="do X", expected_outcome="Y happens")
    assert step.index == 0
    assert step.instruction == "do X"


def test_taktik_defaults():
    t = Taktik(
        strategy_id="str1",
        session_id="s1",
        steps=[TaktikStep(index=0, instruction="do X", expected_outcome="Y")],
    )
    assert t.id
    assert t.required_skills == []
    assert t.estimated_complexity is None
    assert t.judge_verification is None
    assert t.verified is False
    assert t.attempt_number == 1


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------

def test_mission_defaults():
    m = Mission(taktik_id="t1", strategy_id="str1", session_id="s1")
    assert m.id
    assert m.status == "running"
    assert m.attempt_number == 1
    assert m.completed_at is None
    assert m.duration_ms is None
    assert isinstance(m.started_at, datetime)


# ---------------------------------------------------------------------------
# MissionResult
# ---------------------------------------------------------------------------

def test_mission_result_defaults():
    mr = MissionResult(
        mission_id="m1",
        step_index=0,
        action="run tests",
        actual_outcome="tests passed",
        success=True,
    )
    assert mr.id
    assert mr.expected_outcome is None
    assert mr.error_detail is None
    assert mr.artifacts is None
    assert mr.duration_ms is None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def test_memory_defaults():
    mem = Memory(
        type=MemoryType.good_practice,
        scope=MemoryScope.long_term,
        source_agent=AgentName.master,
        content="Always validate inputs",
    )
    assert mem.id
    assert mem.source_session_id is None
    assert mem.structured_content is None
    assert mem.tags == []
    assert mem.relevance_count == 0
    assert mem.last_recalled_at is None
    assert mem.confidence == 1.0
    assert mem.superseded_by is None
    assert mem.expires_at is None
    assert mem.is_active is True


# ---------------------------------------------------------------------------
# AgentNote
# ---------------------------------------------------------------------------

def test_agent_note_defaults():
    note = AgentNote(
        session_id="s1",
        agent_name=AgentName.judge,
        content="Strategy A looks correct",
    )
    assert note.id
    assert note.note_type == NoteType.observation
    assert note.note_references == []
    assert isinstance(note.created_at, datetime)


# ---------------------------------------------------------------------------
# ClientProfile
# ---------------------------------------------------------------------------

def test_client_profile_defaults():
    cp = ClientProfile(client_id="c1")
    assert cp.display_name is None
    assert cp.expertise_level is None
    assert cp.known_domains == []
    assert cp.communication_style is None
    assert cp.preferences == {}
    assert cp.total_sessions == 0
    assert isinstance(cp.first_seen_at, datetime)
    assert isinstance(cp.last_seen_at, datetime)


# ---------------------------------------------------------------------------
# ClientIssueHistory
# ---------------------------------------------------------------------------

def test_client_issue_history_defaults():
    cih = ClientIssueHistory(
        client_id="c1",
        session_id="s1",
        issue_summary="NPE",
        classification=IssueClassification.bug,
        outcome="resolved",
    )
    assert cih.id
    assert cih.winning_strategy_summary is None
    assert cih.jury_score is None


# ---------------------------------------------------------------------------
# StrategyScore
# ---------------------------------------------------------------------------

def test_strategy_score():
    ss = StrategyScore(
        strategy_id="str1",
        session_id="s1",
        correctness=0.9,
        completeness=0.8,
        elegance=0.7,
        robustness=0.85,
        efficiency=0.75,
        weighted_total=0.82,
        reasoning="Solid approach",
    )
    assert ss.id
    assert ss.weighted_total == 0.82
    assert isinstance(ss.created_at, datetime)


# ---------------------------------------------------------------------------
# SessionEvent
# ---------------------------------------------------------------------------

def test_session_event_defaults():
    ev = SessionEvent(
        session_id="s1",
        agent_name=AgentName.reception,
        event_type="session_started",
        payload={"key": "value"},
    )
    assert ev.id is None
    assert ev.phase is None
    assert isinstance(ev.timestamp, datetime)


def test_session_event_with_id():
    ev = SessionEvent(
        id=42,
        session_id="s1",
        agent_name=AgentName.master,
        event_type="llm_call",
        phase="planning",
        payload={},
    )
    assert ev.id == 42
    assert ev.phase == "planning"
