"""Configuration for the SS MCP server."""
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    """Server configuration with sensible defaults."""
    db_path: Path = field(default_factory=lambda: Path("data/ss.db"))
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    num_strategies: int = 3
    max_judge_retries: int = 3
    max_restrategize_rounds: int = 2
    max_concurrent_sampling: int = 1
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
