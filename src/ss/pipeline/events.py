from enum import StrEnum
from typing import Any
from ss.blackboard.models import AgentName, SessionEvent


class EventType(StrEnum):
    SESSION_CREATED = "session_created"
    AGENT_STARTED = "agent_started"
    RECEPTION_INTAKE = "reception_intake"
    MASTER_JOINED = "master_joined"
    STRATEGIES_GENERATED = "strategies_generated"
    TAKTIK_PLANNED = "taktik_planned"
    JUDGE_VERIFIED = "judge_verified"
    JUDGE_REJECTED_LOOP = "judge_rejected_loop"
    MISSION_STARTED = "mission_started"
    MISSION_STEP = "mission_step"
    MISSION_COMPLETED = "mission_completed"
    JURY_SCORED = "jury_scored"
    MASTER_SYNTHESIZED = "master_synthesized"
    MEMORY_CREATED = "memory_created"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"


def create_event(session_id: str, agent: AgentName, event_type: EventType, phase: str | None, payload: dict[str, Any]) -> SessionEvent:
    return SessionEvent(
        session_id=session_id,
        agent_name=agent,
        event_type=str(event_type),
        phase=phase,
        payload=payload,
    )
