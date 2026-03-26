"""Tests for the Repository class."""
from datetime import datetime, timedelta, timezone

import pytest

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
from ss.blackboard.repository import Repository
from ss.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(config: Config) -> Repository:
    db = Database()
    db.connect(config)
    repository = Repository(db)
    yield repository
    db.close()


def _make_session(**kwargs) -> Session:
    defaults = dict(client_id="client1", problem_text="test problem")
    defaults.update(kwargs)
    return Session(**defaults)


def _make_strategy(session_id: str, rank: int = 1, **kwargs) -> Strategy:
    defaults = dict(
        session_id=session_id,
        description="Fix the bug",
        objective="Stability",
        rank=rank,
    )
    defaults.update(kwargs)
    return Strategy(**defaults)


def _make_taktik(strategy_id: str, session_id: str) -> Taktik:
    return Taktik(
        strategy_id=strategy_id,
        session_id=session_id,
        steps=[TaktikStep(index=0, instruction="do X", expected_outcome="Y")],
    )


def _make_mission(taktik_id: str, strategy_id: str, session_id: str) -> Mission:
    return Mission(taktik_id=taktik_id, strategy_id=strategy_id, session_id=session_id)


def _make_memory(**kwargs) -> Memory:
    defaults = dict(
        type=MemoryType.good_practice,
        scope=MemoryScope.long_term,
        source_agent=AgentName.master,
        content="Always validate inputs",
    )
    defaults.update(kwargs)
    return Memory(**defaults)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def test_session_insert_get_roundtrip(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    fetched = repo.get_session(s.id)
    assert fetched is not None
    assert fetched.id == s.id
    assert fetched.client_id == "client1"
    assert fetched.status == SessionStatus.active
    assert fetched.problem_text == "test problem"
    assert fetched.total_llm_calls == 0


def test_get_session_missing(repo: Repository):
    assert repo.get_session("no-such-id") is None


def test_update_session_status(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    now = datetime.now(timezone.utc)
    repo.update_session_status(s.id, SessionStatus.completed, completed_at=now, duration_ms=1000)
    fetched = repo.get_session(s.id)
    assert fetched.status == SessionStatus.completed
    assert fetched.duration_ms == 1000


def test_increment_llm_calls(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    repo.increment_llm_calls(s.id)
    repo.increment_llm_calls(s.id, by=3)
    fetched = repo.get_session(s.id)
    assert fetched.total_llm_calls == 4


def test_list_sessions_filter_by_client(repo: Repository):
    s1 = _make_session(client_id="A")
    s2 = _make_session(client_id="B")
    repo.insert_session(s1)
    repo.insert_session(s2)
    results = repo.list_sessions(client_id="A")
    assert len(results) == 1
    assert results[0].client_id == "A"


def test_list_sessions_filter_by_status(repo: Repository):
    s1 = _make_session()
    s2 = _make_session()
    repo.insert_session(s1)
    repo.insert_session(s2)
    repo.update_session_status(s1.id, SessionStatus.completed)
    active = repo.list_sessions(status=SessionStatus.active)
    assert all(s.status == SessionStatus.active for s in active)
    completed = repo.list_sessions(status=SessionStatus.completed)
    assert len(completed) == 1


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------

def test_issue_insert_get_roundtrip(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    issue = Issue(
        session_id=s.id,
        summary="NPE",
        classification=IssueClassification.bug,
        who="dev",
        where_location="file.py:10",
        why_reason="null ref",
        precondition="null input",
        postcondition="500 error",
        key_points=["validate", "guard"],
        tags=["critical"],
    )
    repo.insert_issue(issue)
    results = repo.get_issue_by_session(s.id)
    assert len(results) == 1
    fetched = results[0]
    assert fetched.summary == "NPE"
    assert fetched.key_points == ["validate", "guard"]
    assert fetched.tags == ["critical"]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def test_strategy_insert_get_roundtrip(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    strat = _make_strategy(s.id)
    repo.insert_strategy(strat)
    fetched = repo.get_strategy(strat.id)
    assert fetched is not None
    assert fetched.description == "Fix the bug"
    assert fetched.status == "planned"


def test_list_strategies_ordered_by_rank(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    s1 = _make_strategy(s.id, rank=2)
    s2 = _make_strategy(s.id, rank=1)
    repo.insert_strategy(s1)
    repo.insert_strategy(s2)
    results = repo.list_strategies(s.id)
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_update_strategy_status(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    strat = _make_strategy(s.id)
    repo.insert_strategy(strat)
    repo.update_strategy_status(strat.id, status="completed", rating_label="excellent", jury_score=0.9)
    fetched = repo.get_strategy(strat.id)
    assert fetched.status == "completed"
    assert fetched.rating_label == "excellent"
    assert fetched.jury_score == pytest.approx(0.9)


def test_update_strategy_score(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    strat = _make_strategy(s.id)
    repo.insert_strategy(strat)
    repo.update_strategy_score(strat.id, jury_score=0.85, jury_metrics={"a": 1})
    fetched = repo.get_strategy(strat.id)
    assert fetched.jury_score == pytest.approx(0.85)
    assert fetched.jury_metrics == {"a": 1}


# ---------------------------------------------------------------------------
# Taktiks
# ---------------------------------------------------------------------------

def test_taktik_insert_verify(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    strat = _make_strategy(s.id)
    repo.insert_strategy(strat)
    taktik = _make_taktik(strat.id, s.id)
    repo.insert_taktik(taktik)
    repo.update_taktik_verification(taktik.id, verified=True, judge_verification={"ok": True})

    # Fetch directly
    row = repo._conn.execute("SELECT * FROM taktiks WHERE id = ?", (taktik.id,)).fetchone()
    assert row["verified"] == 1
    assert row["judge_verification"] is not None


# ---------------------------------------------------------------------------
# Missions
# ---------------------------------------------------------------------------

def test_mission_insert_list(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    strat = _make_strategy(s.id)
    repo.insert_strategy(strat)
    taktik = _make_taktik(strat.id, s.id)
    repo.insert_taktik(taktik)
    mission = _make_mission(taktik.id, strat.id, s.id)
    repo.insert_mission(mission)
    results = repo.list_missions_by_session(s.id)
    assert len(results) == 1
    assert results[0].status == "running"


def test_update_mission_status(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    strat = _make_strategy(s.id)
    repo.insert_strategy(strat)
    taktik = _make_taktik(strat.id, s.id)
    repo.insert_taktik(taktik)
    mission = _make_mission(taktik.id, strat.id, s.id)
    repo.insert_mission(mission)
    now = datetime.now(timezone.utc)
    repo.update_mission_status(mission.id, "completed", completed_at=now, duration_ms=500)
    results = repo.list_missions_by_session(s.id)
    assert results[0].status == "completed"
    assert results[0].duration_ms == 500


# ---------------------------------------------------------------------------
# Mission Results
# ---------------------------------------------------------------------------

def test_mission_results_insert_list(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    strat = _make_strategy(s.id)
    repo.insert_strategy(strat)
    taktik = _make_taktik(strat.id, s.id)
    repo.insert_taktik(taktik)
    mission = _make_mission(taktik.id, strat.id, s.id)
    repo.insert_mission(mission)

    r1 = MissionResult(mission_id=mission.id, step_index=0, action="do A", actual_outcome="A done", success=True)
    r2 = MissionResult(mission_id=mission.id, step_index=1, action="do B", actual_outcome="B failed", success=False, error_detail="oops")
    repo.insert_mission_result(r1)
    repo.insert_mission_result(r2)

    results = repo.list_mission_results(mission.id)
    assert len(results) == 2
    assert results[0].step_index == 0
    assert results[1].success is False
    assert results[1].error_detail == "oops"


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------

def test_memory_insert_get_roundtrip(repo: Repository):
    mem = _make_memory()
    repo.insert_memory(mem)
    fetched = repo.get_memory(mem.id)
    assert fetched is not None
    assert fetched.content == "Always validate inputs"
    assert fetched.is_active is True
    assert fetched.confidence == pytest.approx(1.0)


def test_list_memories_active_only(repo: Repository):
    m1 = _make_memory(content="active one")
    m2 = _make_memory(content="inactive one")
    repo.insert_memory(m1)
    repo.insert_memory(m2)
    repo.supersede_memory(m2.id, superseded_by=m1.id)

    active = repo.list_memories(active_only=True)
    assert all(m.is_active for m in active)
    assert len(active) == 1

    all_mems = repo.list_memories(active_only=False)
    assert len(all_mems) == 2


def test_list_memories_filter_type_scope(repo: Repository):
    m1 = _make_memory(type=MemoryType.good_practice, scope=MemoryScope.long_term)
    m2 = _make_memory(type=MemoryType.bad_practice, scope=MemoryScope.short_term)
    repo.insert_memory(m1)
    repo.insert_memory(m2)

    results = repo.list_memories(type=MemoryType.good_practice)
    assert len(results) == 1
    assert results[0].type == MemoryType.good_practice

    results = repo.list_memories(scope=MemoryScope.short_term)
    assert len(results) == 1
    assert results[0].scope == MemoryScope.short_term


def test_update_memory_confidence(repo: Repository):
    mem = _make_memory()
    repo.insert_memory(mem)
    repo.update_memory_confidence(mem.id, 0.5)
    fetched = repo.get_memory(mem.id)
    assert fetched.confidence == pytest.approx(0.5)


def test_increment_recall(repo: Repository):
    mem = _make_memory()
    repo.insert_memory(mem)
    repo.increment_recall(mem.id)
    repo.increment_recall(mem.id)
    fetched = repo.get_memory(mem.id)
    assert fetched.relevance_count == 2
    assert fetched.last_recalled_at is not None


def test_deactivate_expired(repo: Repository):
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    m_expired = _make_memory(expires_at=past)
    m_future = _make_memory(expires_at=future)
    m_no_expiry = _make_memory()
    repo.insert_memory(m_expired)
    repo.insert_memory(m_future)
    repo.insert_memory(m_no_expiry)

    count = repo.deactivate_expired()
    assert count == 1

    all_mems = repo.list_memories(active_only=False)
    active_ids = {m.id for m in all_mems if m.is_active}
    assert m_expired.id not in active_ids
    assert m_future.id in active_ids
    assert m_no_expiry.id in active_ids


# ---------------------------------------------------------------------------
# Agent Notes
# ---------------------------------------------------------------------------

def test_agent_note_insert_list(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    note = AgentNote(
        session_id=s.id,
        agent_name=AgentName.judge,
        note_type=NoteType.concern,
        content="This might break things",
        note_references=[{"ref": "issue-1"}],
    )
    repo.insert_agent_note(note)
    results = repo.list_agent_notes(s.id)
    assert len(results) == 1
    assert results[0].note_type == NoteType.concern
    assert results[0].note_references == [{"ref": "issue-1"}]


def test_agent_note_filter_by_agent(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    n1 = AgentNote(session_id=s.id, agent_name=AgentName.judge, content="a")
    n2 = AgentNote(session_id=s.id, agent_name=AgentName.master, content="b")
    repo.insert_agent_note(n1)
    repo.insert_agent_note(n2)

    results = repo.list_agent_notes(s.id, agent_name=AgentName.judge)
    assert len(results) == 1
    assert results[0].agent_name == AgentName.judge


# ---------------------------------------------------------------------------
# Session Events
# ---------------------------------------------------------------------------

def test_emit_and_get_events(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    ev1 = SessionEvent(session_id=s.id, agent_name=AgentName.reception, event_type="start", payload={"x": 1})
    ev2 = SessionEvent(session_id=s.id, agent_name=AgentName.master, event_type="plan", payload={"y": 2})

    id1 = repo.emit_event(ev1)
    id2 = repo.emit_event(ev2)
    assert id2 > id1

    events = repo.get_events(s.id)
    assert len(events) == 2
    assert events[0].event_type == "start"
    assert events[1].event_type == "plan"


def test_get_events_after(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    ev1 = SessionEvent(session_id=s.id, agent_name=AgentName.reception, event_type="a", payload={})
    ev2 = SessionEvent(session_id=s.id, agent_name=AgentName.reception, event_type="b", payload={})
    ev3 = SessionEvent(session_id=s.id, agent_name=AgentName.reception, event_type="c", payload={})

    id1 = repo.emit_event(ev1)
    repo.emit_event(ev2)
    repo.emit_event(ev3)

    events = repo.get_events(s.id, after=id1)
    assert len(events) == 2
    assert events[0].event_type == "b"
    assert events[1].event_type == "c"


def test_events_ordered_by_id(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    for i in range(5):
        ev = SessionEvent(session_id=s.id, agent_name=AgentName.reception, event_type=f"e{i}", payload={})
        repo.emit_event(ev)
    events = repo.get_events(s.id)
    ids = [e.id for e in events]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Strategy Scores
# ---------------------------------------------------------------------------

def test_strategy_score_insert(repo: Repository):
    s = _make_session()
    repo.insert_session(s)
    strat = _make_strategy(s.id)
    repo.insert_strategy(strat)
    score = StrategyScore(
        strategy_id=strat.id,
        session_id=s.id,
        correctness=0.9,
        completeness=0.8,
        elegance=0.75,
        robustness=0.85,
        efficiency=0.7,
        weighted_total=0.82,
        reasoning="Solid",
    )
    repo.insert_strategy_score(score)
    row = repo._conn.execute("SELECT * FROM strategy_scores WHERE id = ?", (score.id,)).fetchone()
    assert row is not None
    assert row["weighted_total"] == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# Client Profiles
# ---------------------------------------------------------------------------

def test_upsert_and_get_client_profile(repo: Repository):
    profile = ClientProfile(
        client_id="c1",
        display_name="Alice",
        known_domains=["python", "rust"],
        total_sessions=1,
    )
    repo.upsert_client_profile(profile)
    fetched = repo.get_client_profile("c1")
    assert fetched is not None
    assert fetched.display_name == "Alice"
    assert fetched.known_domains == ["python", "rust"]
    assert fetched.total_sessions == 1


def test_upsert_client_profile_updates(repo: Repository):
    profile = ClientProfile(client_id="c1", total_sessions=1)
    repo.upsert_client_profile(profile)

    updated = ClientProfile(client_id="c1", display_name="Bob", total_sessions=5)
    repo.upsert_client_profile(updated)

    fetched = repo.get_client_profile("c1")
    assert fetched.display_name == "Bob"
    assert fetched.total_sessions == 5


def test_get_client_profile_missing(repo: Repository):
    assert repo.get_client_profile("no-such-client") is None
