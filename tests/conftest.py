"""Shared test fixtures."""
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
from ss.config import Config

@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(db_path=tmp_path / "test.db")

@pytest.fixture
def mock_sampling() -> AsyncMock:
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value="Mock LLM response")
    mock.complete_structured = AsyncMock(return_value={})
    return mock
