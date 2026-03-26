"""FastMCP server entry point for the SS multi-agent orchestrator."""
from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

mcp = FastMCP(
    name="ss",
    instructions=(
        "Multi-agent problem-solving orchestrator. Intake a software engineering "
        "problem and coordinate reception, strategy, planning, judging, mission "
        "execution, jury scoring, and synthesis agents to produce a solution."
    ),
)

# ---------------------------------------------------------------------------
# Lazy-initialized runner
# ---------------------------------------------------------------------------

_runner = None


async def _get_runner():
    """Create and cache the SessionRunner with all dependencies."""
    global _runner
    if _runner is not None:
        return _runner

    from ss.blackboard.database import Database
    from ss.blackboard.repository import Repository
    from ss.config import Config
    from ss.memory.manager import MemoryManager
    from ss.pipeline.runner import SessionRunner
    from ss.sampling.adapter import SamplingAdapter
    from ss.vectors.encoder import EmbeddingEncoder
    from ss.vectors.store import VectorStore

    config = Config()

    db = Database()
    db.connect(config)

    repo = Repository(db)

    encoder = EmbeddingEncoder()
    vector_store = VectorStore(db, encoder)

    memory_mgr = MemoryManager(repo, vector_store, config)

    # SamplingAdapter requires an MCP context for real sampling.
    # When running as an MCP server, the context is provided at call time
    # via the lifespan/request context. We pass None here as a placeholder;
    # in production the adapter is wired to the live MCP context.
    sampling = SamplingAdapter(mcp_context=None, max_concurrent=config.max_concurrent_sampling)

    _runner = SessionRunner(
        repo=repo,
        vector_store=vector_store,
        memory_mgr=memory_mgr,
        sampling=sampling,
        config=config,
    )

    return _runner


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def solve(problem: str, context: dict | None = None) -> dict:
    """Submit a software engineering problem for multi-agent analysis and solving.

    Returns a session_id that can be used to poll for results and events.
    """
    runner = await _get_runner()
    session_id = await runner.start(problem, context)
    return {"session_id": session_id, "status": "started"}


@mcp.tool()
async def get_events(session_id: str, after: int = 0) -> list[dict]:
    """Retrieve pipeline events for a session, optionally after a given event id."""
    runner = await _get_runner()
    return await runner.get_events(session_id, after=after)


@mcp.tool()
async def get_result(session_id: str) -> dict:
    """Get the synthesized result for a completed session."""
    runner = await _get_runner()
    return await runner.get_result(session_id)


@mcp.tool()
async def cancel(session_id: str) -> dict:
    """Cancel an active session pipeline."""
    runner = await _get_runner()
    return await runner.cancel(session_id)


@mcp.tool()
async def recall(query: str, type: str | None = None, limit: int = 5) -> list[dict]:
    """Search the agent memory store for relevant past learnings.

    Args:
        query: The search query text.
        type: Optional memory type filter (e.g. 'good_practice', 'pattern').
        limit: Maximum number of results to return.
    """
    runner = await _get_runner()
    from ss.blackboard.models import MemoryType

    mem_type = MemoryType(type) if type else None
    memories = runner._memory_mgr.recall(query, type=mem_type, limit=limit)
    return [m.model_dump(mode="json") for m in memories]


@mcp.tool()
async def get_session_history(client_id: str | None = None, limit: int = 10) -> list[dict]:
    """Return recent session history, optionally filtered by client id."""
    runner = await _get_runner()
    return await runner.get_history(client_id=client_id, limit=limit)


@mcp.tool()
async def inspect_session(session_id: str) -> dict:
    """Return a full snapshot of a session including issues, strategies, missions."""
    runner = await _get_runner()
    return await runner.inspect(session_id)


@mcp.tool()
async def get_agent_notes(session_id: str, agent_name: str | None = None) -> list[dict]:
    """Return agent notes for a session, optionally filtered by agent name."""
    runner = await _get_runner()
    return await runner.get_notes(session_id, agent_name=agent_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
