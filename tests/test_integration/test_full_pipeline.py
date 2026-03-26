"""End-to-end integration tests for the full SS pipeline."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ss.blackboard.database import Database
from ss.blackboard.repository import Repository
from ss.config import Config
from ss.memory.manager import MemoryManager
from ss.pipeline.runner import SessionRunner
from ss.sampling.adapter import SamplingAdapter
from ss.vectors.encoder import EmbeddingEncoder
from ss.vectors.store import VectorStore


# ---------------------------------------------------------------------------
# Smart mock sampling adapter
# ---------------------------------------------------------------------------

def _make_issue_response() -> dict:
    return {
        "summary": "Memory leak in API handler causing OOM crashes",
        "classification": "bug",
        "severity": "high",
        "who": "Backend engineers",
        "where_location": "API request handler in server.py",
        "why_reason": "Connection objects not released after request completion",
        "precondition": "Server running under moderate load",
        "postcondition": "No memory leak, stable RSS over time",
        "key_points": [
            "Connection pool not cleaned up",
            "OOM kills occur after ~4 hours",
            "RSS grows linearly with request count",
        ],
    }


def _make_client_assessment_response() -> dict:
    return {
        "expertise_level": "advanced",
        "domains": ["backend", "python", "performance"],
        "communication_style": "technical",
    }


def _make_strategies_response() -> dict:
    return {
        "strategies": [
            {
                "description": "Fix connection pool lifecycle management",
                "objective": "Ensure connections are released after each request",
                "approach_type": "bug_fix",
                "confidence": 0.9,
                "rank": 1,
            },
            {
                "description": "Implement connection timeout and recycling",
                "objective": "Automatically recycle stale connections",
                "approach_type": "refactor",
                "confidence": 0.8,
                "rank": 2,
            },
            {
                "description": "Add memory profiling and automated leak detection",
                "objective": "Detect and alert on memory leaks in CI",
                "approach_type": "testing",
                "confidence": 0.7,
                "rank": 3,
            },
        ]
    }


def _make_taktik_response() -> dict:
    return {
        "steps": [
            {
                "index": 0,
                "instruction": "Audit the connection pool initialization and teardown",
                "expected_outcome": "List of unclosed connection paths identified",
            },
            {
                "index": 1,
                "instruction": "Add context manager wrapping around all connection usage",
                "expected_outcome": "Connections closed automatically on exit",
            },
        ],
        "required_skills": ["Python", "asyncio", "connection pooling"],
        "estimated_complexity": "medium",
    }


def _make_judge_verified_response() -> dict:
    return {
        "verified": True,
        "reasoning": "Plan correctly addresses connection lifecycle",
        "issues_found": [],
        "suggestions": ["Consider adding tests for edge cases"],
    }


def _make_mission_step_response() -> dict:
    return {
        "action": "Audited connection pool and found 3 unclosed paths",
        "actual_outcome": "All connection paths now properly closed",
        "success": True,
    }


def _make_jury_scoring_response() -> dict:
    return {
        "correctness": 0.9,
        "completeness": 0.85,
        "elegance": 0.8,
        "robustness": 0.88,
        "efficiency": 0.82,
        "reasoning": "Solution effectively fixes the memory leak with proper cleanup",
    }


def _make_synthesis_response() -> dict:
    return {
        "final_answer": (
            "Fix the memory leak by wrapping all connection usage in context managers "
            "and auditing the connection pool teardown logic. The root cause is that "
            "connection objects are not released after requests complete."
        ),
        "key_insights": [
            "Connection pool not cleaned up properly",
            "Context managers ensure deterministic cleanup",
        ],
        "recommendations": [
            "Add connection pool monitoring",
            "Write regression tests for resource cleanup",
        ],
    }


def _make_distribute_learnings_response() -> dict:
    """Response for distribute_learnings call (plain text via llm_call, not structured)."""
    return {}


class SmartMockSamplingAdapter(SamplingAdapter):
    """A SamplingAdapter that returns appropriate JSON based on the agent's system prompt."""

    def __init__(self) -> None:
        super().__init__(mcp_context=None, max_concurrent=10)
        self._call_count = 0

    async def _do_sample(self, **kwargs: Any) -> str:
        system_prompt: str = kwargs.get("system_prompt", "")
        self._call_count += 1

        # Detect agent by distinctive phrases in the system prompt
        if "Reception Agent" in system_prompt:
            return json.dumps(_make_issue_response())

        if "Master Agent" in system_prompt:
            # Two modes: join_session (assess client) or synthesize (final answer)
            messages = kwargs.get("messages", [])
            user_content = messages[0]["content"] if messages else ""
            if "winning strategy" in user_content.lower() or "Winning strategy" in user_content:
                return json.dumps(_make_synthesis_response())
            else:
                return json.dumps(_make_client_assessment_response())

        if "Strategist Agent" in system_prompt:
            return json.dumps(_make_strategies_response())

        if "Taktik Planner Agent" in system_prompt:
            return json.dumps(_make_taktik_response())

        if "Judge Agent" in system_prompt:
            return json.dumps(_make_judge_verified_response())

        if "Mission Agent" in system_prompt:
            return json.dumps(_make_mission_step_response())

        if "Jury Agent" in system_prompt:
            return json.dumps(_make_jury_scoring_response())

        # Fallback: return empty JSON object
        return "{}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        db_path=tmp_path / "integration_test.db",
        num_strategies=3,
        max_judge_retries=2,
        max_restrategize_rounds=1,
        max_concurrent_sampling=5,
    )


@pytest.fixture
def db(config: Config):
    database = Database()
    database.connect(config)
    yield database
    database.close()


@pytest.fixture
def repo(db: Database) -> Repository:
    return Repository(db)


@pytest.fixture
def encoder() -> EmbeddingEncoder:
    return EmbeddingEncoder()


@pytest.fixture
def vector_store(db: Database, encoder: EmbeddingEncoder) -> VectorStore:
    return VectorStore(db, encoder)


@pytest.fixture
def memory_mgr(repo: Repository, vector_store: VectorStore, config: Config) -> MemoryManager:
    return MemoryManager(repo, vector_store, config)


@pytest.fixture
def sampling() -> SmartMockSamplingAdapter:
    return SmartMockSamplingAdapter()


@pytest.fixture
def runner(
    repo: Repository,
    vector_store: VectorStore,
    memory_mgr: MemoryManager,
    sampling: SmartMockSamplingAdapter,
    config: Config,
) -> SessionRunner:
    return SessionRunner(
        repo=repo,
        vector_store=vector_store,
        memory_mgr=memory_mgr,
        sampling=sampling,
        config=config,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _wait_for_completion(runner: SessionRunner, session_id: str, timeout: float = 30.0) -> dict:
    """Poll get_result until the session is no longer active."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        result = await runner.get_result(session_id)
        status = result.get("status", "running")
        if status != "running":
            return result
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"Session {session_id} did not complete within {timeout}s")
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_completes(runner: SessionRunner) -> None:
    """Full pipeline should run end-to-end and produce a completed result."""
    session_id = await runner.start(
        problem="Our API server has a memory leak that causes OOM crashes after 4 hours.",
        client_id="test-client",
    )
    assert session_id is not None

    result = await _wait_for_completion(runner, session_id)

    assert result["status"] == "completed", f"Unexpected status: {result}"


@pytest.mark.asyncio
async def test_full_pipeline_has_synthesis(runner: SessionRunner) -> None:
    """Completed session should have a synthesis payload with an answer."""
    session_id = await runner.start(
        problem="Our API server has a memory leak that causes OOM crashes after 4 hours.",
        client_id="test-client",
    )
    result = await _wait_for_completion(runner, session_id)

    assert "synthesis" in result, f"No synthesis in result: {result}"
    synthesis = result["synthesis"]
    # The synthesis event payload contains final_answer_preview at minimum
    assert synthesis is not None


@pytest.mark.asyncio
async def test_full_pipeline_events_include_required_types(runner: SessionRunner) -> None:
    """Pipeline should emit all expected event types."""
    session_id = await runner.start(
        problem="Our API server has a memory leak that causes OOM crashes after 4 hours.",
        client_id="test-client",
    )
    await _wait_for_completion(runner, session_id)

    events = await runner.get_events(session_id)
    event_types = {e["event_type"] for e in events}

    required_events = {
        "session_created",
        "reception_intake",
        "strategies_generated",
        "jury_scored",
        "master_synthesized",
    }
    for expected in required_events:
        assert expected in event_types, (
            f"Missing event type '{expected}'. Got: {event_types}"
        )


@pytest.mark.asyncio
async def test_full_pipeline_inspect_returns_issue_strategies_missions(runner: SessionRunner) -> None:
    """inspect_session should return issue, strategies, and missions."""
    session_id = await runner.start(
        problem="Our API server has a memory leak that causes OOM crashes after 4 hours.",
        client_id="test-client",
    )
    await _wait_for_completion(runner, session_id)

    snapshot = await runner.inspect(session_id)

    assert "issues" in snapshot
    assert len(snapshot["issues"]) >= 1, "Expected at least one issue"

    assert "strategies" in snapshot
    assert len(snapshot["strategies"]) >= 1, "Expected at least one strategy"

    assert "missions" in snapshot
    assert len(snapshot["missions"]) >= 1, "Expected at least one mission"


@pytest.mark.asyncio
async def test_cancel_session(runner: SessionRunner) -> None:
    """Cancelling an active session should mark it as cancelled."""
    session_id = await runner.start(
        problem="Cancel test: this session should be cancelled.",
        client_id="cancel-client",
    )

    # Cancel immediately (task may or may not still be running)
    cancel_result = await runner.cancel(session_id)
    assert cancel_result["status"] == "cancelled"
    assert cancel_result["session_id"] == session_id

    # get_result should reflect cancelled status
    result = await runner.get_result(session_id)
    assert result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_session_history(runner: SessionRunner) -> None:
    """get_history should return sessions for a given client."""
    client_id = "history-test-client"

    # Run two sessions for the same client
    session_id_1 = await runner.start(
        problem="First problem for history test.",
        client_id=client_id,
    )
    session_id_2 = await runner.start(
        problem="Second problem for history test.",
        client_id=client_id,
    )

    # Wait for both to complete
    await _wait_for_completion(runner, session_id_1)
    await _wait_for_completion(runner, session_id_2)

    history = await runner.get_history(client_id=client_id, limit=10)

    assert len(history) >= 2
    session_ids_in_history = {s["id"] for s in history}
    assert session_id_1 in session_ids_in_history
    assert session_id_2 in session_ids_in_history


@pytest.mark.asyncio
async def test_memories_created_after_session(
    runner: SessionRunner,
    memory_mgr: MemoryManager,
) -> None:
    """After a session completes, memories should be stored and recallable."""
    problem = "Memory leak in connection pool needs fixing with context managers."

    session_id = await runner.start(problem=problem, client_id="memory-test-client")
    await _wait_for_completion(runner, session_id)

    # Query memories related to the problem
    memories = memory_mgr.recall(query="memory leak connection pool", limit=10)
    assert len(memories) >= 1, (
        "Expected at least one memory to be stored after session completion"
    )


@pytest.mark.asyncio
async def test_get_result_unknown_session(runner: SessionRunner) -> None:
    """get_result for an unknown session_id should return an error."""
    result = await runner.get_result("nonexistent-session-id")
    assert "error" in result


@pytest.mark.asyncio
async def test_inspect_unknown_session(runner: SessionRunner) -> None:
    """inspect for an unknown session_id should return an error."""
    result = await runner.inspect("nonexistent-session-id")
    assert "error" in result


@pytest.mark.asyncio
async def test_multiple_concurrent_sessions(runner: SessionRunner) -> None:
    """Multiple sessions can run concurrently and all complete."""
    problems = [
        "Fix the authentication bug in the login flow.",
        "Optimize the database query that takes 30 seconds.",
        "Refactor the legacy payment module to use the new API.",
    ]

    session_ids = await asyncio.gather(
        *[
            runner.start(problem=p, client_id=f"concurrent-client-{i}")
            for i, p in enumerate(problems)
        ]
    )

    # Wait for all to complete
    results = await asyncio.gather(
        *[_wait_for_completion(runner, sid) for sid in session_ids]
    )

    for i, result in enumerate(results):
        assert result["status"] == "completed", (
            f"Session {i} did not complete: {result}"
        )


@pytest.mark.asyncio
async def test_session_has_duration_ms(runner: SessionRunner) -> None:
    """Completed session result should include duration_ms."""
    session_id = await runner.start(
        problem="Test duration tracking.",
        client_id="duration-client",
    )
    result = await _wait_for_completion(runner, session_id)

    assert "duration_ms" in result
    assert result["duration_ms"] is not None
    assert result["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_events_pagination_after_parameter(runner: SessionRunner) -> None:
    """get_events with after parameter should return only events after the given id."""
    session_id = await runner.start(
        problem="Test event pagination.",
        client_id="pagination-client",
    )
    await _wait_for_completion(runner, session_id)

    all_events = await runner.get_events(session_id)
    assert len(all_events) > 0

    # Get events after the first event's id
    first_event_id = all_events[0]["id"]
    later_events = await runner.get_events(session_id, after=first_event_id)

    assert len(later_events) == len(all_events) - 1
    for e in later_events:
        assert e["id"] > first_event_id
