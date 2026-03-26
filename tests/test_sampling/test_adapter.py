"""Tests for SamplingAdapter."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ss.sampling.adapter import SamplingAdapter


# ---------------------------------------------------------------------------
# Subclass to override _do_sample without needing real MCP context
# ---------------------------------------------------------------------------

class MockSamplingAdapter(SamplingAdapter):
    """Test double that records call order and returns configurable responses."""

    def __init__(self, responses: list[str], max_concurrent: int = 1):
        super().__init__(mcp_context=None, max_concurrent=max_concurrent)
        self._responses = list(responses)
        self._call_log: list[str] = []

    async def _do_sample(self, **kwargs: Any) -> str:
        response = self._responses.pop(0)
        return response


# ---------------------------------------------------------------------------
# _parse_json static method
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_raw_json_object(self):
        text = '{"key": "value", "num": 42}'
        result = SamplingAdapter._parse_json(text)
        assert result == {"key": "value", "num": 42}

    def test_markdown_fenced_json(self):
        text = '```json\n{"answer": true}\n```'
        result = SamplingAdapter._parse_json(text)
        assert result == {"answer": True}

    def test_markdown_fenced_with_surrounding_text(self):
        text = "Here is the result:\n```json\n{\"status\": \"ok\"}\n```\nDone."
        result = SamplingAdapter._parse_json(text)
        assert result == {"status": "ok"}

    def test_raises_on_invalid(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            SamplingAdapter._parse_json("this is not json at all")

    def test_raises_on_non_json_in_fence(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            SamplingAdapter._parse_json("```json\nnot valid json\n```")

    def test_nested_json(self):
        text = '{"nested": {"list": [1, 2, 3]}}'
        result = SamplingAdapter._parse_json(text)
        assert result == {"nested": {"list": [1, 2, 3]}}


# ---------------------------------------------------------------------------
# Semaphore serialization
# ---------------------------------------------------------------------------

class TestSemaphoreSerializtion:
    @pytest.mark.asyncio
    async def test_max_concurrent_1_serializes_calls(self):
        """With max_concurrent=1, two tasks must execute start/end/start/end."""
        events: list[str] = []
        barrier = asyncio.Event()

        class TrackingAdapter(SamplingAdapter):
            def __init__(self):
                super().__init__(mcp_context=None, max_concurrent=1)

            async def _do_sample(self, **kwargs: Any) -> str:
                events.append("start")
                await asyncio.sleep(0.01)
                events.append("end")
                return "response"

        adapter = TrackingAdapter()

        async def call():
            return await adapter.complete("sys", [{"role": "user", "content": "hi"}])

        await asyncio.gather(call(), call())

        # With serialization: start, end, start, end (not start, start, end, end)
        assert events == ["start", "end", "start", "end"]

    @pytest.mark.asyncio
    async def test_max_concurrent_2_allows_parallel(self):
        """With max_concurrent=2, two tasks should both start before either ends."""
        events: list[str] = []
        both_started = asyncio.Event()

        class TrackingAdapter(SamplingAdapter):
            def __init__(self):
                super().__init__(mcp_context=None, max_concurrent=2)
                self._start_count = 0

            async def _do_sample(self, **kwargs: Any) -> str:
                self._start_count += 1
                events.append("start")
                if self._start_count == 2:
                    both_started.set()
                await both_started.wait()
                await asyncio.sleep(0.001)
                events.append("end")
                return "response"

        adapter = TrackingAdapter()

        async def call():
            return await adapter.complete("sys", [{"role": "user", "content": "hi"}])

        await asyncio.gather(call(), call())
        assert events[:2] == ["start", "start"]


# ---------------------------------------------------------------------------
# complete_structured
# ---------------------------------------------------------------------------

class TestCompleteStructured:
    @pytest.mark.asyncio
    async def test_complete_structured_parses_json_response(self):
        response = '{"result": "success", "value": 42}'
        adapter = MockSamplingAdapter(responses=[response])
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        result = await adapter.complete_structured(
            system_prompt="Return JSON",
            messages=[{"role": "user", "content": "give me JSON"}],
            schema=schema,
        )
        assert result == {"result": "success", "value": 42}

    @pytest.mark.asyncio
    async def test_complete_structured_parses_fenced_json(self):
        response = '```json\n{"status": "done"}\n```'
        adapter = MockSamplingAdapter(responses=[response])
        schema = {}
        result = await adapter.complete_structured(
            system_prompt="Return JSON",
            messages=[{"role": "user", "content": "go"}],
            schema=schema,
        )
        assert result == {"status": "done"}

    @pytest.mark.asyncio
    async def test_complete_structured_schema_in_prompt(self):
        """Schema dict should appear in the enhanced system prompt."""
        captured_kwargs: list[dict] = []

        class CapturingAdapter(SamplingAdapter):
            def __init__(self):
                super().__init__(mcp_context=None, max_concurrent=1)

            async def _do_sample(self, **kwargs: Any) -> str:
                captured_kwargs.append(kwargs)
                return '{"ok": true}'

        adapter = CapturingAdapter()
        schema = {"type": "object", "required": ["answer"]}
        await adapter.complete_structured(
            system_prompt="base prompt",
            messages=[{"role": "user", "content": "hi"}],
            schema=schema,
        )
        system_used = captured_kwargs[0].get("system_prompt", "")
        assert "JSON" in system_used or "json" in system_used


# ---------------------------------------------------------------------------
# _do_sample with None context raises RuntimeError
# ---------------------------------------------------------------------------

class TestDoSampleGuard:
    @pytest.mark.asyncio
    async def test_do_sample_raises_without_context(self):
        adapter = SamplingAdapter(mcp_context=None)
        with pytest.raises(RuntimeError, match="mcp_context"):
            await adapter._do_sample(system_prompt="x", messages=[])
