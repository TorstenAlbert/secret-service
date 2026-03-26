# SS

<p align="center">
  <img src="https://img.shields.io/badge/version-0.0.1-blue?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/pipeline_agents-7-green?style=for-the-badge" alt="7 Pipeline Agents">
  <img src="https://img.shields.io/badge/scoring_dimensions-5-purple?style=for-the-badge" alt="5 Scoring Dimensions">
  <img src="https://img.shields.io/badge/python-3.11+-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MCP-FastMCP_3.x-flat?style=flat-square" alt="FastMCP">
  <img src="https://img.shields.io/badge/models-Pydantic_v2-flat?style=flat-square" alt="Pydantic">
  <img src="https://img.shields.io/badge/storage-SQLite_+_vss-flat?style=flat-square" alt="SQLite + vss">
  <img src="https://img.shields.io/badge/embeddings-MiniLM_384d-flat?style=flat-square" alt="Embeddings">
  <img src="https://img.shields.io/badge/tests-239_passing-flat?style=flat-square&color=brightgreen" alt="Tests">
</p>

A multi-agent MCP server that acts as a meta-orchestration layer between the client and the user. Problems flow through 7 specialized LLM-backed agents that analyse, strategise, plan, verify, execute, and score solutions in parallel competing branches. Failed approaches are remembered as anti-patterns; winning tactics become reusable patterns across sessions.

## Architecture

Parallel strategy fan-out with blackboard coordination:

```
                       ┌───────────────────────────────────────────────┐
                       │  1. INTAKE                                     │
                       │     Reception → Issue classification            │
                       │     Master    → Client identity                 │
                       │     Strategist → N competing strategies         │
                       └───────────────────────┬───────────────────────┘
                                               │
                  ┌────────────────────────────▼────────────────────────────┐
                  │  2. PARALLEL STRATEGY FAN-OUT                           │
                  │                                                         │
                  │  ┌─ Strategy 1: Taktik Planner → Judge → Mission ─┐    │
                  │  ├─ Strategy 2: Taktik Planner → Judge → Mission ─┤    │
                  │  └─ Strategy 3: Taktik Planner → Judge → Mission ─┘    │
                  │                                                         │
                  │  Judge rejects? → Retry within branch (max 3×)          │
                  │  All branches fail? → Re-strategise (max 2 rounds)      │
                  └────────────────────────────┬───────────────────────────┘
                                               │
                       ┌───────────────────────▼───────────────────────┐
                       │  3. EVALUATION                                 │
                       │     Jury   → 5-dimension scoring per mission   │
                       │     Master → Synthesise final answer            │
                       │     Master → Distribute learnings (memories)    │
                       └───────────────────────────────────────────────┘

            All agents read/write to a shared BLACKBOARD (SQLite + sqlite-vss)
```

## Key Concepts

### The 7 Agents

| Agent | Temp | Role | Reads | Writes |
|-------|------|------|-------|--------|
| **Reception** | 0.1 | Intake, classify, structure | Client input, past memories | Issue, Session |
| **Master** | 0.3 | Orchestrate, teach, synthesise | All notes and knowledge | Identity, learnings, answer |
| **Strategist** | 0.9 | Creative strategy generation | Issue, long-term memories | N ranked Strategies |
| **Taktik Planner** | 0.8 | Step-by-step execution plan | Strategy, short-term memories | Taktik with steps + skills |
| **Judge** | 0.1 | Verify for errors and gaps | Taktik, postcondition | Verification, error notes |
| **Mission** | 0.2 | Execute steps, record results | Verified Taktik | Step-by-step results |
| **Jury** | 0.2 | Score and compare missions | All results, Judge notes | 5-dim scores, rankings |

### 5-Dimension Scoring

| Dimension | Weight | Dimension | Weight |
|-----------|--------|-----------|--------|
| Correctness | 30% | Elegance | 15% |
| Completeness | 25% | Efficiency | 10% |
| Robustness | 20% | | |

### Memory System

Two brains with three scopes:

| Scope | Lifetime | Teaches | Contains |
|-------|----------|---------|----------|
| **Short-term** | Current session | Taktik Planner | Attempt results, observations |
| **Long-term** | Persists forever | Strategist | Good/bad practices |
| **Permanent** | Never decays | Strategist | Proven patterns, anti-patterns |

Memory features: embedding-based recall (sqlite-vss, 384-dim), confidence decay on contradiction, near-duplicate supersession, relevance tracking.

### Strategy Lifecycle

```
  PLANNED ──branch starts──▸ IN_PROGRESS ──mission succeeds──▸ SUCCEEDED
                                  │                                │
                                  ├── mission fails ──▸ FAILED     │
                                  │                                │
                             Jury scores:                     Jury scores:
                             score ≥ 0.75 → "proven"          score < 0.30 → "archived"
                             score ≥ 0.30 → "adequate"
```

Score thresholds drive memory creation: proven strategies become reusable `pattern` memories; archived strategies become `anti_pattern` warnings.

### Issue Classification

Every problem is structured by the Reception Agent into:

| Field | Purpose |
|-------|---------|
| `who` | Who has the problem |
| `where_location` | Where it is happening |
| `why_reason` | Why it is happening |
| `precondition` | True before the problem |
| `postcondition` | True after the problem is solved |
| `classification` | bug, architecture, performance, refactor, security, testing, deployment, documentation |
| `severity` | critical, high, medium, low, info |
| `key_points` | Step-by-step breakdown |

### LLM vs. Deterministic Boundary

| LLM does | Code does |
|----------|-----------|
| Problem decomposition | State management (SQLite) |
| Strategy generation | Pipeline orchestration |
| Taktik planning | Parallel branch execution |
| Verification critique | Score aggregation & weighting |
| Mission execution | Memory lifecycle (decay, supersession) |
| Performance evaluation | Event streaming & observability |
| Output synthesis | Vector indexing (sqlite-vss) |
| | Client profile tracking |

### Client Identity

The Master Agent builds a profile of each client across sessions:

| Field | Example |
|-------|---------|
| `expertise_level` | beginner, intermediate, advanced, expert |
| `known_domains` | ["python", "devops", "frontend"] |
| `communication_style` | concise, detailed, visual |
| `total_sessions` | 42 |

The Reception Agent uses this to tailor intake; the Master Agent uses it to calibrate the final answer.

## Installation

```bash
git clone <repo-url>
cd ss
pip install -e ".[dev]"
```

### Run

```bash
# stdio transport (for Claude Code)
ss

# or explicitly
python -m ss.server
```

### Configure in Claude Code

```json
{
  "mcpServers": {
    "ss": {
      "command": "/path/to/ss/.venv/bin/python",
      "args": ["-m", "ss.server"]
    }
  }
}
```

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `db_path` | `data/ss.db` | SQLite database location |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `embedding_dimension` | `384` | Vector dimensions |
| `num_strategies` | `3` | Strategies per session |
| `max_judge_retries` | `3` | Judge rejection retries per branch |
| `max_restrategize_rounds` | `2` | Full re-strategise rounds |
| `max_concurrent_sampling` | `1` | Concurrent MCP sampling calls |
| `similarity_threshold` | `0.15` | Distance for memory supersession |
| `confidence_decay_rate` | `0.05` | Per-contradiction confidence loss |
| `proven_threshold` | `0.75` | Score to label strategy "proven" |
| `archive_threshold` | `0.30` | Score to label strategy "archived" |

### Run Tests

```bash
pytest tests/ -v   # 239 tests
```

## MCP Interface

<details>
<summary><b>Session Lifecycle Tools (4)</b></summary>

| Tool | Description |
|------|-------------|
| `solve` | Start a new problem-solving session. Pipeline runs async. Returns `session_id` immediately. |
| `get_events` | Poll for pipeline events after a given event ID. Use for streaming progress. |
| `get_result` | Get the final result of a completed session. |
| `cancel` | Cancel a running session. |

</details>

<details>
<summary><b>Memory & Learning Tools (2)</b></summary>

| Tool | Description |
|------|-------------|
| `recall` | Search memories by semantic similarity. Filter by type: `good_practice`, `bad_practice`, `pattern`, `anti_pattern`, `knowledge`, `insight`. |
| `get_session_history` | List past sessions, optionally filtered by client. |

</details>

<details>
<summary><b>Observability Tools (2)</b></summary>

| Tool | Description |
|------|-------------|
| `inspect_session` | Full session trace: issue, strategies, taktiks, missions, scores, notes, memories. |
| `get_agent_notes` | Notes written by agents during a session. Filter by agent: `reception`, `master`, `strategist`, `taktik_planner`, `judge`, `mission`, `jury`. |

</details>

### Event Types (16)

<details>
<summary><b>Streaming Events</b></summary>

| Event | Payload |
|-------|---------|
| `session_created` | `{ session_id, problem_summary }` |
| `agent_started` | `{ agent_name, phase }` |
| `reception_intake` | `{ issue_summary, classification, who, where, why }` |
| `master_joined` | `{ client_identity }` |
| `strategies_generated` | `{ count, strategies: [{ id, description, rank }] }` |
| `taktik_planned` | `{ strategy_id, steps_count, required_skills }` |
| `judge_verified` | `{ taktik_id, verified, rejection_reason? }` |
| `judge_rejected_loop` | `{ taktik_id, attempt, max_attempts, reason }` |
| `mission_started` | `{ strategy_id, mission_id }` |
| `mission_step` | `{ mission_id, step_index, action, outcome }` |
| `mission_completed` | `{ mission_id, status }` |
| `jury_scored` | `{ scores: [{ strategy_id, score, metrics }] }` |
| `master_synthesized` | `{ winning_strategy_id, final_answer }` |
| `memory_created` | `{ type, scope, content_preview }` |
| `session_completed` | `{ session_id, status, total_events, duration_ms }` |
| `session_failed` | `{ session_id, error, last_agent }` |

</details>

## Pipeline Call Budget

| Phase | Calls | Parallelism |
|-------|-------|-------------|
| Intake (Reception + Master + Strategist) | 3 | Sequential |
| Execution (Taktik + Judge + Mission) × N strategies | 3 × N | Parallel branches |
| Evaluation (Jury + Master synthesis) | 2 | Sequential |
| **Total for 3 strategies** | **14** | |

Worst case with retries: 2 rounds × 3 strategies × 3 judge retries = 18 taktik attempts.

## Data Model

```
Session ─┬── Issue
         ├── Strategy ─┬── Taktik ── Mission ── MissionResult
         │             └── StrategyScore
         ├── SessionEvent (append-only stream)
         └── AgentNote

Memory (good_practice | bad_practice | pattern | anti_pattern | knowledge | insight)
ClientProfile ── ClientIssueHistory

EmbeddingRegistry ←→ vss_* virtual tables (sqlite-vss, 384-dim)
```

Every content-bearing entity is indexed in sqlite-vss for embedding-based similarity search.

## Project Structure

```
src/ss/
├── server.py                     # FastMCP server — 8 tools
├── config.py                     # Configuration dataclass
├── blackboard/
│   ├── database.py               # SQLite connection, WAL mode, sqlite-vss, migrations
│   ├── schema.sql                # Full DDL — 13 tables, 14 indexes, 6 vss tables
│   ├── models.py                 # Pydantic models (Session, Issue, Strategy, Taktik, ...)
│   └── repository.py             # CRUD operations per entity type
├── vectors/
│   ├── encoder.py                # Lazy-loading SentenceTransformer wrapper
│   └── store.py                  # VectorStore with sqlite-vss (index, search, find_similar)
├── memory/
│   ├── manager.py                # Store, recall, supersede, distribute learnings
│   ├── client_profile.py         # Client identity tracking across sessions
│   └── cleanup.py                # Expiry, confidence decay, garbage collection
├── agents/
│   ├── base.py                   # BaseAgent ABC — LLM calls, events, notes, memory recall
│   ├── reception.py              # Reception Agent (temp 0.1) — intake, classify, structure
│   ├── master.py                 # Master Agent (temp 0.3) — orchestrate, teach, synthesise
│   ├── strategist.py             # Strategist Agent (temp 0.9) — creative strategy generation
│   ├── taktik_planner.py         # Taktik Planner Agent (temp 0.8) — step-by-step planning
│   ├── judge.py                  # Judge Agent (temp 0.1) — verify for errors and gaps
│   ├── mission.py                # Mission Agent (temp 0.2) — execute and record results
│   └── jury.py                   # Jury Agent (temp 0.2) — score and compare missions
├── pipeline/
│   ├── runner.py                 # SessionRunner — full pipeline orchestration
│   ├── branch.py                 # StrategyBranch — parallel Taktik → Judge → Mission
│   └── events.py                 # EventType enum, event creation helpers
└── sampling/
    └── adapter.py                # MCP sampling wrapper with semaphore concurrency control

tests/
├── test_blackboard/              # Models, database, repository (59 tests)
├── test_vectors/                 # Encoder, vector store (41 tests)
├── test_sampling/                # Sampling adapter (12 tests)
├── test_memory/                  # Manager, cleanup (29 tests)
├── test_agents/                  # Base agent + all 7 agents (82 tests)
├── test_pipeline/                # Events, branch, runner (16 tests)
└── test_integration/             # End-to-end pipeline (12 tests — full 239 total)
```

## Roadmap

| Phase | Status |
|-------|--------|
| 1 — Core Pipeline (7 agents, blackboard, parallel fan-out) | **Complete** |
| 2 — Memory System (sqlite-vss, decay, supersession, learning) | **Complete** |
| 3 — Observability (event streaming, session inspection, agent notes) | **Complete** |
| 4 — Direct LLM Provider (fallback when MCP sampling unavailable) | Planned |
| 5 — Dashboard (web UI for session traces, memory browser, metrics) | Planned |
| 6 — Skill Discovery (awesome-agent-skills-mcp integration in Taktik Planner) | Planned |

## Star History

<a href="https://www.star-history.com/?repos=TorstenAlbert%2Fsecret-service&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=TorstenAlbert/secret-service&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=TorstenAlbert/secret-service&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=TorstenAlbert/secret-service&type=date&legend=top-left" />
 </picture>
</a>

## License

See [LICENSE](LICENSE) for details.
