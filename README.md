<h1 align="center">ss</h1>

<p align="center">
  <strong>A multi-agent MCP server that solves software-engineering problems through a council of competing strategies — with project-level memory that learns across sessions.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/protocol-MCP%20(FastMCP)-6E40C9?style=flat-square" alt="MCP / FastMCP">
  <img src="https://img.shields.io/badge/tests-350%20passing-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/lint-ruff%20clean-46a2f1?style=flat-square&logo=ruff&logoColor=white" alt="Ruff">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/models-Pydantic_v2-E92063?style=flat-square&logo=pydantic&logoColor=white" alt="Pydantic v2">
  <img src="https://img.shields.io/badge/storage-SQLite_+_sqlite--vss-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite + sqlite-vss">
  <img src="https://img.shields.io/badge/embeddings-MiniLM_384d-FF6F00?style=flat-square" alt="Embeddings">
  <img src="https://img.shields.io/badge/transport-stdio-555?style=flat-square" alt="stdio">
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-key-concepts">Concepts</a> ·
  <a href="#%EF%B8%8F-configuration">Configuration</a> ·
  <a href="#-mcp-interface">MCP Interface</a>
</p>

---

`ss` is a meta-orchestration layer that sits between an MCP client (e.g. Claude Code) and a hard problem. Instead of a single model answering in one shot, a problem flows through a **council of LLM personas** that propose and peer-review competing strategies, then through **parallel agentic branches** that plan, verify, execute, and score solutions in a bounded loop. Failed approaches are remembered as anti-patterns; winning tactics become reusable patterns — and a persistent **Project Memory Layer** records *why* the project is the way it is, so the next session starts smarter.

> **Tested MCP clients:** Claude Code (stdio).

## Table of Contents

- [Highlights](#-highlights)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Key Concepts](#-key-concepts)
- [Configuration](#%EF%B8%8F-configuration)
- [MCP Interface](#-mcp-interface)
- [Pipeline Call Budget](#-pipeline-call-budget)
- [Data Model & Project Structure](#-data-model)
- [Roadmap](#-roadmap)
- [License](#-license)

## ✨ Highlights

- **🧑‍⚖️ Strategy Council** — five distinct thinking lenses (risk, first-principles, ambition, outsider, pragmatist) propose strategies, then peer-review each other **anonymously** before a neutral Chair synthesises ranked options. Convergence-stopped to save calls.
- **🔁 Bounded agentic loop** — each strategy runs an `observe → plan → verify → act → check → decide` loop with hard stops (`max_loop_iterations`, `no_progress_threshold`), so it always terminates.
- **🧠 Project Memory Layer (PML)** — a local, deterministic store of `STATE / INTENT / DECISION / WHY / TIMELINE / HEALTH` that answers *"why is the project the way it is?"* and grounds every session.
- **🗂️ Code Index Layer (CIL)** — an `ast`-based identifier index for token-efficient navigation: `summary`, `signatures`, `query`, change-detection, notes, tasks, and a local live-log reader.
- **🧬 Cross-session learning** — proven patterns and anti-patterns persist in an embedding-backed memory (sqlite-vss) and feed the next council round.
- **🔌 Pluggable Model Router** — one configurable model per agent via OpenRouter (free default model), with MCP-sampling fallback. One-call `solve_sync` for non-sampling harnesses.

## 🚀 Quick Start

```bash
# 1. Install
git clone <repo-url> && cd ss
pip install -e ".[dev]"

# 2. Point the Model Router at a provider (free default model — no credits needed)
export OPENROUTER_API_KEY=sk-or-v1-...

# 3. Run the MCP server (stdio)
ss            # == python -m ss.server
```

**Register in Claude Code** (`~/.claude/mcp.json` or your client's MCP config):

```json
{
  "mcpServers": {
    "ss": {
      "command": "/path/to/ss/.venv/bin/python",
      "args": ["-m", "ss.server"],
      "env": { "OPENROUTER_API_KEY": "sk-or-v1-..." }
    }
  }
}
```

Then call the **`solve_sync`** tool with a problem statement to get a synthesized answer in a single round-trip, or **`solve`** for the blocking-with-soft-timeout variant. Without a key the server still starts and read-only tools work, but `solve` returns a `NoProvider` error (see [Provider Workflow](#provider-workflow)).

```bash
pytest tests/ -v          # full suite — 350 tests
ruff check . && ruff format .
```

## 🧭 Architecture

`Reception → LLM Council → parallel BranchLoops → Jury → Master`, over a shared SQLite blackboard, with PML/CIL providing project-level context:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. INTAKE                                                          │
  │     Reception  → Issue classification  (reads PML state / health)   │
  │     Master     → Client identity (cross-session profile)            │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────────────┐
  │  2. LLM COUNCIL  (persona layer, grounded by PML)                   │
  │                                                                     │
  │  Stage 1 — Propose (parallel, one voice per persona):               │
  │    risk_analyst · first_principles · ambition · naive_outsider ·    │
  │    pragmatist                                                       │
  │                                                                     │
  │  Stage 2 — Anonymised peer review (parallel × rounds,               │
  │            anti-pattern injection, convergence-stopped)             │
  │                                                                     │
  │  Stage 3 — Chairman synthesis → N ranked strategies                 │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────────────┐
  │  3. PARALLEL BRANCH LOOPS  (one per strategy, concurrent)           │
  │                                                                     │
  │  For each strategy:                                                 │
  │    observe → plan (CIL signatures + find-skills) → verify (Judge) → │
  │    act (Mission, CIL query) → check (PostconditionChecker) → decide │
  │    ↑__________________________|  (bounded by max_loop_iterations    │
  │                                  and no_progress_threshold)         │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────────────┐
  │  4. EVALUATION & LEARNING                                           │
  │     Jury   → 5-dimension scoring per mission                        │
  │     Master → Synthesise final answer                                │
  │     Master → Cross-session learning → blackboard patterns +         │
  │              PML decision/why/timeline (fed into next session)      │
  └─────────────────────────────────────────────────────────────────────┘

          All agents read/write a shared BLACKBOARD (SQLite + sqlite-vss)
```

## 🧠 Key Concepts

### LLM Council

The Council replaces a single strategist with a **persona-driven debate** before any branch work begins. Five thinking lenses deliberate and cross-review; a neutral Chair synthesises ranked strategies.

| Persona key | Name | Thinking lens | Proposal temp |
|-------------|------|---------------|:---:|
| `risk_analyst` | Risk Analyst | Hunt failure modes; name the single biggest risk the other voices will overlook. | 0.5 |
| `first_principles` | First Principles | Question root assumptions; strip the problem to fundamentals and rebuild. | 0.7 |
| `ambition` | Ambition | Find the 10× approach — the most ambitious path that is still genuinely viable. | 0.9 |
| `naive_outsider` | Naive Outsider | Assume zero domain context; surface the curse of knowledge insiders miss. | 0.8 |
| `pragmatist` | Pragmatist | Find the smallest executable first step; if it can't ship today, it isn't an answer. | 0.4 |
| `chairman` | Chairman | Neutral synthesis into ranked strategies — note agreement, clashes, and blind spots. | 0.3 |

Each persona shapes one member's **proposal** (Stage 1) and its **review lens** (Stage 2). Identities are anonymised before peer review so members evaluate on merit; peer review always runs at temperature `0.3` regardless of proposal temp. The Chair never proposes. Configure with `council_personas` (trim to reduce cost) and `council_persona_temps`; models are cycled across personas via `AgentModelConfig.council_members`, so even one configured model still applies all lenses.

### Agentic Loop (BranchLoop)

Each strategy branch runs a bounded **observe → plan → verify → act → check → decide** loop rather than a single-shot pass. Hard stops guarantee termination:

| Control | Default | Effect |
|---------|:---:|--------|
| `max_loop_iterations` | 10 | Absolute cap on loop iterations per branch |
| `no_progress_threshold` | 3 | Consecutive non-improving iterations escalate to re-strategise |
| `max_judge_retries` | 3 | Inner judge-retry loop per plan step |
| `loop_token_budget` | `None` | Optional token ceiling across a branch |

`PostconditionChecker` gates the **check** phase and prefers mechanical evidence from CIL's live log over LLM guessing.

### find-skills Pre-pass

Before planning, the Taktik Planner discovers installable skills via the [`vercel-labs/skills`](https://github.com/vercel-labs/skills) CLI — it runs `npx skills add <source> --list` against `findskills_source` (default `vercel-labs/agent-skills`), ranks the results against the strategy, and injects the top `findskills_max_results` into the plan prompt (recorded in `BranchResult.required_skills` for cross-session learning). It is **best-effort**: if `npx`/node is unavailable or the call times out, discovery quietly no-ops and never blocks the pipeline. The listing is cached per source, so the CLI runs at most once per session. Toggle with `findskills_enabled`.

### The Agents

| Agent | Temp | Role | Reads | Writes |
|-------|:---:|------|-------|--------|
| **Reception** | 0.1 | Intake, classify, structure | Client input, memories, PML state/health | Issue, Session |
| **Master** | 0.3 | Orchestrate, teach, synthesise | All notes & knowledge | Identity, learnings, answer, PML |
| **LLM Council** | varies | Persona debate → N strategies | Issue, anti-patterns, PML intent/decisions | N ranked Strategies |
| **Taktik Planner** | 0.8 | Step-by-step execution plan | Strategy, memories, CIL signatures, skills | Taktik with steps + skills |
| **Judge** | 0.1 | Verify for errors, gaps, decision violations | Taktik, postcondition, PML decisions | Verification, error notes |
| **Mission** | 0.2 | Execute steps, record results | Verified Taktik, CIL query | Step results, CIL notes |
| **Jury** | 0.2 | Score and compare missions | All results, Judge notes | 5-dim scores, rankings |

### 5-Dimension Scoring

| Dimension | Weight | | Dimension | Weight |
|-----------|:---:|---|-----------|:---:|
| Correctness | 30% | | Elegance | 15% |
| Completeness | 25% | | Efficiency | 10% |
| Robustness | 20% | | | |

### Memory System

Embedding-backed memory with three scopes:

| Scope | Lifetime | Consumed by | Contains |
|-------|----------|-------------|----------|
| **Short-term** | Current session | Taktik Planner | Attempt results, observations |
| **Long-term** | Persists | LLM Council | Good / bad practices |
| **Permanent** | Never decays | LLM Council | Proven patterns, anti-patterns |

Features: embedding-based recall (sqlite-vss, 384-dim), confidence decay on contradiction, near-duplicate supersession, relevance tracking.

### Project Memory Layer (PML)

A local, deterministic (LLM-free) store that answers **"why is the project the way it is?"** across sessions. Stored under `pml_dir` (default `.project-memory/`); toggle with `pml_enabled` (degrades to a no-op when disabled).

| Record type | Purpose |
|-------------|---------|
| `STATE` | Current observable state of the project |
| `INTENT` | Goals and direction |
| `DECISION` | Choices made and their rationale |
| `WHY` | Root-cause explanations |
| `TIMELINE` | Sequenced events and milestones |
| `HEALTH` | Quality signals, risks, blockers |

**Operations:** `capture(type, content)` to write, `inspect(type)` to read, `as_context(types)` to format for prompt injection — plus convenience wrappers (`capture_decision/why/timeline/note`, `state()/health()/decisions()/intent()/timeline()`).

**Agent integration:** Reception reads STATE & HEALTH; Council reads INTENT, DECISION & TIMELINE; Judge enforces DECISION as binding constraints; Master writes DECISION, WHY & TIMELINE after synthesis.

### Code Index Layer (CIL)

A local, `ast`-based identifier index for **token-efficient code navigation** (no subprocess, no language server). Stored under `cil_dir` (default `.code-index/`), indexing `cil_index_root` (default `src`).

| Operation | Description |
|-----------|-------------|
| `summary()` | File count, identifier count, breakdown by kind |
| `signatures(path)` | Functions/classes in a file (or path prefix), no file read |
| `query(term, mode)` | Identifier lookup — `contains` / `starts_with` / `exact` |
| `session(path)` | Change detection + incremental reindex of modified files |
| `note(path, text)` | Attach / read notes for a file path (survives sessions) |
| `task(action)` | Open / update / list lightweight follow-up tasks |
| `log(action)` | Read a **local** NDJSON log file (no HTTP) — filter by `level` / `since` |

**Agent integration:** PLAN reads `signatures`; ACT runs `query` and writes `note`; CHECK reads `log` for execution evidence; OBSERVE runs `session` to detect changes between loop iterations. CIL lowers token cost *per call* — the loop budget governs the *number* of calls.

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

Proven strategies become reusable `pattern` memories; archived strategies become `anti_pattern` warnings — both feed the next Council round.

### Issue Classification

Reception structures every problem into:

| Field | Purpose |
|-------|---------|
| `who` | Who has the problem |
| `where_location` | Where it is happening |
| `why_reason` | Why it is happening |
| `precondition` | True before the problem |
| `postcondition` | True after the problem is solved |
| `classification` | bug · architecture · performance · refactor · security · testing · deployment · documentation |
| `severity` | critical · high · medium · low · info |
| `key_points` | Step-by-step breakdown |

### LLM vs. Deterministic Boundary

| LLM does | Code does |
|----------|-----------|
| Problem decomposition | State management (SQLite) |
| Strategy generation & peer review | Pipeline orchestration & parallelism |
| Taktik planning | Score aggregation & weighting |
| Verification critique | Memory lifecycle (decay, supersession) |
| Mission execution | Event streaming & observability |
| Output synthesis | Vector indexing (sqlite-vss), code indexing (`ast`) |

## ⚙️ Configuration

### LLM Provider

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

The default model is `nvidia/nemotron-3-super-120b-a12b:free` (free tier — no credits, but a free API key is required). Paid models are opt-in per agent via `AgentModelConfig`.

### Python API

```python
from ss.config import AgentModelConfig, Config

config = Config(
    # Model Router (OpenRouter)
    openrouter_api_key="sk-or-v1-...",                  # or set OPENROUTER_API_KEY
    openrouter_default_model="nvidia/nemotron-3-super-120b-a12b:free",
    openrouter_agent_models=AgentModelConfig(
        # Per-agent overrides; None falls back to openrouter_default_model.
        # e.g. council_chairman="anthropic/claude-sonnet-4-5",
        #      judge="anthropic/claude-sonnet-4-5",
    ),

    # LLM Council
    council_personas=["risk_analyst", "first_principles", "ambition",
                      "naive_outsider", "pragmatist"],
    council_persona_temps={"ambition": 0.95},           # optional per-persona temp
    council_review_rounds=3,                            # max peer-review rounds
    council_convergence_threshold=0.85,                 # stop early when reviews converge

    # Agentic loop (per branch)
    max_loop_iterations=10,
    no_progress_threshold=3,
    loop_token_budget=None,

    # find-skills (discovery via the `npx skills` CLI)
    findskills_enabled=True,
    findskills_source="vercel-labs/agent-skills",
    findskills_max_results=5,

    # Project Memory Layer / Code Index Layer
    pml_enabled=True, pml_dir=".project-memory",
    cil_enabled=True, cil_dir=".code-index", cil_index_root="src",

    # solve() blocking
    solve_wait_timeout=90.0,
)
```

<details>
<summary><b>Full Config Reference</b></summary>

| Setting | Default | Description |
|---------|---------|-------------|
| `db_path` | `data/ss.db` | SQLite database location |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `embedding_dimension` | `384` | Vector dimensions |
| `num_strategies` | `3` | Strategies per session |
| `max_judge_retries` | `3` | Judge rejection retries per branch iteration |
| `max_restrategize_rounds` | `2` | Full re-strategise rounds when all branches fail |
| `max_concurrent_sampling` | `8` | Concurrent LLM calls (semaphore) |
| `similarity_threshold` | `0.15` | Distance for memory supersession |
| `confidence_decay_rate` | `0.05` | Per-contradiction confidence loss |
| `proven_threshold` | `0.75` | Score to label strategy "proven" |
| `archive_threshold` | `0.30` | Score to label strategy "archived" |
| `openrouter_default_model` | `…nemotron…:free` | Default Model Router model |
| `council_personas` | 5 lenses | Persona keys; trim to reduce council cost |
| `council_persona_temps` | `{}` | Per-persona temperature overrides |
| `council_review_rounds` | `3` | Max peer-review rounds (convergence may stop earlier) |
| `council_convergence_threshold` | `0.85` | Mean pairwise cosine to declare convergence |
| `max_loop_iterations` | `10` | Hard cap on agentic loop iterations per branch |
| `no_progress_threshold` | `3` | Stagnant iterations before a branch escalates |
| `loop_token_budget` | `None` | Optional token ceiling per branch |
| `findskills_enabled` | `True` | Enable find-skills pre-pass (`npx skills` CLI) |
| `findskills_source` | `vercel-labs/agent-skills` | Source repo for `npx skills add … --list` |
| `findskills_max_results` | `5` | Max skills surfaced into the plan prompt |
| `pml_enabled` / `pml_dir` | `True` / `.project-memory` | Project Memory Layer toggle + store dir |
| `cil_enabled` / `cil_dir` | `True` / `.code-index` | Code Index Layer toggle + store dir |
| `cil_index_root` | `src` | Root directory CIL indexes |
| `cil_log_file` | `.code-index/live.log` | Local NDJSON log read by `CIL.log()` |
| `solve_wait_timeout` | `90.0` | Seconds `solve()` blocks before returning a poll handle |

</details>

## 🔌 MCP Interface

**Transport:** stdio only (no HTTP/SSE). `health` is exposed as an **MCP tool**, not an HTTP route. The list tools (`get_events`, `recall`, `get_session_history`, `get_agent_notes`) return `{"message": "..."}` when empty rather than an empty array.

<details>
<summary><b>Session Lifecycle Tools (5)</b></summary>

| Tool | Description |
|------|-------------|
| `solve` | Submit a problem. Blocks by default (`wait=True`); returns the synthesized answer, or a `session_id` to poll if it exceeds `solve_wait_timeout`. |
| `solve_sync` | Submit a problem and block until the pipeline finishes with **no soft timeout** — final answer in one call. For non-sampling harnesses or long-tolerant clients. |
| `get_events` | Poll pipeline events after a given event ID (streaming progress). |
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
<summary><b>Observability Tools (3)</b></summary>

| Tool | Description |
|------|-------------|
| `health` | Liveness probe — `{"status": "ok", "timestamp": ...}`. |
| `inspect_session` | Full session trace: issue, strategies, taktiks, missions, scores, notes, memories. |
| `get_agent_notes` | Notes written by agents during a session; filter by agent name. |

</details>

<details>
<summary><b>Event Types (16)</b></summary>

`session_created` · `agent_started` · `reception_intake` · `master_joined` · `strategies_generated` · `taktik_planned` · `judge_verified` · `judge_rejected_loop` · `mission_started` · `mission_step` · `mission_completed` · `jury_scored` · `master_synthesized` · `memory_created` · `session_completed` · `session_failed`

`SessionEvent` rows are an append-only stream; `get_result` reconstructs the answer by scanning backwards for `master_synthesized`.

</details>

### Provider Workflow

```
  OPENROUTER_API_KEY set?
    yes → OpenRouter            (preferred; free default model costs nothing)
    no  → MCP client sampling available?
            yes → MCP sampling fallback
                  (most clients — incl. Claude Code — don't implement it;
                   deprecated as of MCP spec 2026-07-28)
            no  → disabled: solve()/solve_sync() return {"code": "NoProvider", ...}
                  read-only tools (recall, get_session_history, …) still work
```

### `solve` Contract

`solve` is **blocking by default** (`wait=True`):

| Parameter | Behaviour |
|-----------|-----------|
| `wait=True` *(default)* | Blocks until the pipeline completes and returns the synthesized answer. On exceeding `solve_wait_timeout` (90 s), returns `{"session_id", "status": "running", "message"}` while the pipeline keeps running. |
| `wait=False` | Returns immediately with `{"session_id", "status": "started"}`. |

`solve_sync` always runs to completion (no soft timeout). Set `solve_wait_timeout` below your MCP client's tool-call timeout to avoid client-side cutoffs.

## 📊 Pipeline Call Budget

| Phase | Calls | Parallelism |
|-------|-------|-------------|
| Intake (Reception + Master) | 2 | Sequential |
| Council Stage 1 — Proposals | P personas | Parallel |
| Council Stage 2 — Peer review | P × rounds (convergence-stopped) | Parallel |
| Council Stage 3 — Chair synthesis | 1 | — |
| Branch loops (Taktik + Judge + Mission) | 3 × N strategies × L iterations | Parallel branches |
| Evaluation (Jury + Master synthesis) | 2 | Sequential |

With the default **P = 5 personas** and a single review round, the council alone is `5 + 5 + 1 = 11` calls.
**Worked example** (5 personas, 1 review round, 3 strategies, 3 loop iterations): `2 + 11 + (3 × 3 × 3) + 2 = 42` calls.

**Cost caps:** `max_loop_iterations`, `no_progress_threshold`, `loop_token_budget`, and Stage-2 convergence early-exit. Trim `council_personas` to reduce council cost proportionally.

## 🗂️ Data Model

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

Session/agent state lives in the **blackboard** (SQLite + sqlite-vss). The **PML** (`.project-memory/`) and **CIL** (`.code-index/`) are separate local stores with their own SQLite files.

<details>
<summary><b>Project Structure</b></summary>

```
src/ss/
├── server.py                 # FastMCP server — 10 MCP tools
├── config.py                 # Config + AgentModelConfig dataclasses
├── blackboard/               # SQLite blackboard: database, schema.sql, models, repository
├── vectors/                  # encoder (MiniLM) + VectorStore (sqlite-vss)
├── memory/
│   ├── manager.py            # store, recall, supersede, distribute learnings
│   ├── project_memory.py     # Project Memory Layer (PML)
│   ├── client_profile.py     # client identity across sessions
│   └── cleanup.py            # expiry, confidence decay, GC
├── intel/
│   └── code_index.py         # Code Index Layer (CIL) — ast index + log reader
├── agents/
│   ├── base.py · reception.py · master.py · taktik_planner.py
│   ├── judge.py · mission.py · jury.py · strategist.py (legacy)
│   └── council/              # LLMCouncil, member, chairman, personas
├── loop/                     # branch_loop, postcondition, progress
├── pipeline/                 # runner (orchestration), branch (legacy), events
├── sampling/                 # adapter, openrouter (Model Router), per_agent, factory
└── skills/                   # finder, resolver (find-skills)

tests/   # 350 tests, mirroring src/ss/<package>/ under tests/test_<package>/
```

</details>

## 🗺️ Roadmap

| Phase | Status |
|-------|:---:|
| 1 — Core Pipeline (agents, blackboard, parallel fan-out) | ✅ Complete |
| 2 — Memory System (sqlite-vss, decay, supersession, learning) | ✅ Complete |
| 3 — Observability (event streaming, session inspection, agent notes) | ✅ Complete |
| 4 — LLM Council (persona layer, convergence, cross-session learning) | ✅ Complete |
| 5 — Agentic Loop (BranchLoop, PostconditionChecker, find-skills) | ✅ Complete |
| 6 — Project Intelligence (PML, CIL, Model Router, `solve_sync`) | ✅ Complete |
| 7 — Dashboard (web UI: session traces, memory browser, metrics) | 🔜 Planned |

## 📄 License

Released under the [MIT License](LICENSE) — © 2026 Torsten Albert.

## ⭐ Star History

<a href="https://www.star-history.com/?repos=TorstenAlbert%2Fsecret-service&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=TorstenAlbert/secret-service&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=TorstenAlbert/secret-service&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=TorstenAlbert/secret-service&type=date&legend=top-left" />
 </picture>
</a>
