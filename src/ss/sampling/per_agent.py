"""Resolve the Model Router model id for a given agent/persona."""
from __future__ import annotations

from ss.config import Config

HARD_DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def resolve_agent_model(config: Config, agent_name: str) -> str:
    """Per-agent override → openrouter_default_model → hard default."""
    override = getattr(config.openrouter_agent_models, agent_name, None)
    return override or config.openrouter_default_model or HARD_DEFAULT_MODEL
