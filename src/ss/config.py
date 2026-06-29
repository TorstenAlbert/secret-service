"""Configuration for the SS MCP server."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentModelConfig:
    """Per-agent OpenRouter model overrides. ``None`` = use the default model."""
    reception: str | None = None
    master: str | None = None
    council_members: list[str] = field(default_factory=lambda: [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ])
    council_chairman: str = "nvidia/nemotron-3-super-120b-a12b:free"  # e.g. "anthropic/claude-sonnet-4-5"
    taktik_planner: str | None = None
    judge: str | None = None
    mission: str | None = None
    jury: str | None = None


@dataclass
class Config:
    """Server configuration with sensible defaults."""
    db_path: Path = field(default_factory=lambda: Path("data/ss.db"))
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    num_strategies: int = 3
    max_judge_retries: int = 3
    max_restrategize_rounds: int = 2
    max_concurrent_sampling: int = 8  # raised from 1 so Council fan-out + parallel branches actually run concurrently
    similarity_threshold: float = 0.15
    confidence_decay_rate: float = 0.05
    proven_threshold: float = 0.75
    archive_threshold: float = 0.3
    score_weights: dict[str, float] = field(default_factory=lambda: {
        "correctness": 0.30,
        "completeness": 0.25,
        "robustness": 0.20,
        "elegance": 0.15,
        "efficiency": 0.10,
    })

    # OpenRouter
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_default_model: str = "nvidia/nemotron-3-super-120b-a12b:free"  # e.g. "anthropic/claude-sonnet-4-5"
    openrouter_timeout: float = 120.0
    openrouter_site_url: str = ""
    openrouter_site_name: str = "secret-service"
    openrouter_agent_models: AgentModelConfig = field(default_factory=AgentModelConfig)

    # LLM Council
    council_anonymise_identities: bool = True
    council_review_rounds: int = 3
    council_convergence_threshold: float = 0.85  # mean pairwise review cosine >= this -> stop early

    # Agentic loop (per branch)
    max_loop_iterations: int = 10
    no_progress_threshold: int = 3
    loop_token_budget: int | None = None

    # LLM Council persona layer
    council_personas: list[str] = field(default_factory=lambda: [
        "risk_analyst", "first_principles", "ambition", "naive_outsider", "pragmatist",
    ])
    council_persona_temps: dict[str, float] = field(default_factory=dict)

    # find-skills (discovery via the `npx skills` CLI)
    findskills_enabled: bool = True
    findskills_source: str = "vercel-labs/agent-skills"  # `npx skills add <source> --list`
    findskills_max_results: int = 5  # cap skills surfaced into the plan prompt

    # Project Memory Layer (PML)
    pml_enabled: bool = True
    pml_dir: str = ".project-memory"

    # Code Index Layer (CIL)
    cil_enabled: bool = True
    cil_dir: str = ".code-index"
    cil_index_root: str = "src"
    cil_log_file: str = ".code-index/live.log"

    # solve() blocking behaviour
    solve_wait_timeout: float = 90.0  # seconds solve() blocks before returning a poll-me response

    def __post_init__(self) -> None:
        if self.openrouter_api_key is None:
            env_key = os.environ.get("OPENROUTER_API_KEY")
            if env_key:
                self.openrouter_api_key = env_key

    def model_for_agent(self, agent_name: str) -> str:
        """Resolve the OpenRouter model id for a pipeline agent."""
        override = getattr(self.openrouter_agent_models, agent_name, None)
        return override or self.openrouter_default_model
