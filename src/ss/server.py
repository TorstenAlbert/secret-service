"""FastMCP server entry point for the SS multi-agent orchestrator."""
from __future__ import annotations

import logging
import time

from fastmcp import Context, FastMCP

mcp = FastMCP(
    name="ss",
    instructions=(
        "Multi-agent problem-solving orchestrator. Intake a software engineering "
        "problem and coordinate reception, strategy, planning, judging, mission "
        "execution, jury scoring, and synthesis agents to produce a solution."
    ),
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _epoch_ms() -> int:
    return int(time.time() * 1000)


def _mask(secret: str | None) -> str:
    if not secret:
        return "<unset>"
    return f"***{secret[-4:]}" if len(secret) > 4 else "***"


def _nonempty(items: list, desc: str):
    if not items:
        return {"message": f"No results found for: '{desc}'. Try different keywords."}
    return items


def _error(exc: Exception, session_id: str | None = None) -> dict:
    logger.exception("tool error: %s", type(exc).__name__)
    return {"error": str(exc), "code": type(exc).__name__, "session_id": session_id}


def _log_call(tool: str, session_id: str | None, t0: float, outcome: str) -> None:
    logger.info(
        "tool=%s session_id=%s duration_ms=%d outcome=%s",
        tool, session_id, int((time.time() - t0) * 1000), outcome,
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
    from ss.sampling.factory import build_adapter
    from ss.vectors.encoder import EmbeddingEncoder
    from ss.vectors.store import VectorStore

    config = Config()

    db = Database()
    db.connect(config)

    repo = Repository(db)

    encoder = EmbeddingEncoder()
    vector_store = VectorStore(db, encoder)

    memory_mgr = MemoryManager(repo, vector_store, config)

    bundle = build_adapter(config)
    if not bundle.uses_openrouter:
        logger.warning(
            "Running without OpenRouter (key=%s); LLM sampling disabled — solve() will return a config error.",
            _mask(config.openrouter_api_key),
        )

    _runner = SessionRunner(
        repo=repo,
        vector_store=vector_store,
        memory_mgr=memory_mgr,
        bundle=bundle,
        config=config,
    )

    return _runner


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def health() -> dict:
    """Liveness probe. Returns service status and a millisecond timestamp."""
    return {"status": "ok", "timestamp": _epoch_ms(), "service": "secret-service"}


@mcp.tool()
async def solve(
    problem: str,
    context: dict | None = None,
    wait: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Submit a software-engineering problem for multi-agent solving.

    Blocks until the pipeline finishes and returns the synthesized result. If it
    exceeds the configured soft timeout, returns a session_id to poll get_result
    with. Pass wait=False to return immediately with a session_id.
    """
    # `ctx` is injected by FastMCP and excluded from the tool's input schema.
    t0 = time.time()
    try:
        runner = await _get_runner()
        # Attach the live MCP context so the sampling fallback can work when no
        # OpenRouter key is set (no-op in OpenRouter mode).
        if ctx is not None:
            runner.bind_context(ctx)
        if not runner.llm_available:
            return {
                "error": (
                    "No LLM provider available. Set OPENROUTER_API_KEY, or use an MCP "
                    "client that supports sampling. (Claude Code does not implement sampling.)"
                ),
                "code": "NoProvider",
                "session_id": None,
            }
        return await runner.solve(problem, context=context, wait=wait)
    except Exception as exc:
        return _error(exc, None)
    finally:
        _log_call("solve", None, t0, "done")


@mcp.tool()
async def solve_sync(
    problem: str,
    context: dict | None = None,
    ctx: Context | None = None,
) -> dict:
    """Submit a problem and block until the pipeline finishes, returning the final answer.

    Unlike ``solve``, this tool does not apply a soft timeout — it awaits pipeline
    completion in a single call. Use it when the MCP client can tolerate long-running
    tool invocations (e.g. in test harnesses or non-sampling environments).
    """
    t0 = time.time()
    try:
        runner = await _get_runner()
        if ctx is not None:
            runner.bind_context(ctx)
        if not runner.llm_available:
            return {
                "error": (
                    "No LLM provider available. Set OPENROUTER_API_KEY, or use an MCP "
                    "client that supports sampling. (Claude Code does not implement sampling.)"
                ),
                "code": "NoProvider",
                "session_id": None,
            }
        return await runner.solve(problem, context=context, wait=True, timeout=None)
    except Exception as exc:
        return _error(exc, None)
    finally:
        _log_call("solve_sync", None, t0, "done")


@mcp.tool()
async def get_events(session_id: str, after: int = 0) -> list[dict] | dict:
    """Retrieve pipeline events for a session, optionally after a given event id."""
    t0 = time.time()
    try:
        runner = await _get_runner()
        events = await runner.get_events(session_id, after=after)
        return _nonempty(events, f"events for session '{session_id}' after {after}")
    except Exception as exc:
        return _error(exc, session_id)
    finally:
        _log_call("get_events", session_id, t0, "done")


@mcp.tool()
async def get_result(session_id: str) -> dict:
    """Get the synthesized result for a completed session."""
    t0 = time.time()
    try:
        runner = await _get_runner()
        return await runner.get_result(session_id)
    except Exception as exc:
        return _error(exc, session_id)
    finally:
        _log_call("get_result", session_id, t0, "done")


@mcp.tool()
async def cancel(session_id: str) -> dict:
    """Cancel an active session pipeline."""
    t0 = time.time()
    try:
        runner = await _get_runner()
        return await runner.cancel(session_id)
    except Exception as exc:
        return _error(exc, session_id)
    finally:
        _log_call("cancel", session_id, t0, "done")


@mcp.tool()
async def recall(query: str, type: str | None = None, limit: int = 5) -> list[dict] | dict:
    """Search the agent memory store for relevant past learnings.

    Args:
        query: The search query text.
        type: Optional memory type filter (e.g. 'good_practice', 'pattern').
        limit: Maximum number of results to return.
    """
    t0 = time.time()
    try:
        runner = await _get_runner()
        from ss.blackboard.models import MemoryType

        mem_type = MemoryType(type) if type else None
        memories = runner._memory_mgr.recall(query, type=mem_type, limit=limit)
        memories_as_dicts = [m.model_dump(mode="json") for m in memories]
        return _nonempty(memories_as_dicts, query)
    except Exception as exc:
        return _error(exc, None)
    finally:
        _log_call("recall", None, t0, "done")


@mcp.tool()
async def get_session_history(client_id: str | None = None, limit: int = 10) -> list[dict] | dict:
    """Return recent session history, optionally filtered by client id."""
    t0 = time.time()
    try:
        runner = await _get_runner()
        history = await runner.get_history(client_id=client_id, limit=limit)
        return _nonempty(history, client_id or "all clients")
    except Exception as exc:
        return _error(exc, None)
    finally:
        _log_call("get_session_history", None, t0, "done")


@mcp.tool()
async def inspect_session(session_id: str) -> dict:
    """Return a full snapshot of a session including issues, strategies, missions."""
    t0 = time.time()
    try:
        runner = await _get_runner()
        return await runner.inspect(session_id)
    except Exception as exc:
        return _error(exc, session_id)
    finally:
        _log_call("inspect_session", session_id, t0, "done")


@mcp.tool()
async def get_agent_notes(session_id: str, agent_name: str | None = None) -> list[dict] | dict:
    """Return agent notes for a session, optionally filtered by agent name."""
    t0 = time.time()
    try:
        runner = await _get_runner()
        notes = await runner.get_notes(session_id, agent_name=agent_name)
        return _nonempty(notes, f"notes for session '{session_id}'")
    except Exception as exc:
        return _error(exc, session_id)
    finally:
        _log_call("get_agent_notes", session_id, t0, "done")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
