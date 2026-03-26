"""Tests for pipeline events module."""
import pytest
from ss.pipeline.events import EventType, create_event
from ss.blackboard.models import AgentName, SessionEvent


def test_all_16_event_types_exist():
    expected = [
        "session_created",
        "agent_started",
        "reception_intake",
        "master_joined",
        "strategies_generated",
        "taktik_planned",
        "judge_verified",
        "judge_rejected_loop",
        "mission_started",
        "mission_step",
        "mission_completed",
        "jury_scored",
        "master_synthesized",
        "memory_created",
        "session_completed",
        "session_failed",
    ]
    event_values = [str(e) for e in EventType]
    assert len(event_values) == 16
    for name in expected:
        assert name in event_values, f"Missing event type: {name}"


def test_event_type_is_str_enum():
    assert isinstance(EventType.SESSION_CREATED, str)
    assert EventType.SESSION_CREATED == "session_created"


def test_create_event_produces_session_event():
    session_id = "sess-123"
    agent = AgentName.reception
    event_type = EventType.RECEPTION_INTAKE
    phase = "intake"
    payload = {"problem": "something broken"}

    event = create_event(session_id, agent, event_type, phase, payload)

    assert isinstance(event, SessionEvent)
    assert event.session_id == session_id
    assert event.agent_name == agent
    assert event.event_type == "reception_intake"
    assert event.phase == phase
    assert event.payload == payload


def test_create_event_with_none_phase():
    event = create_event("sid", AgentName.master, EventType.SESSION_CREATED, None, {})
    assert event.phase is None
    assert event.event_type == "session_created"


def test_create_event_payload_preserved():
    payload = {"key": "value", "count": 42, "nested": {"a": 1}}
    event = create_event("sid", AgentName.judge, EventType.JUDGE_VERIFIED, "verify", payload)
    assert event.payload == payload
