"""Sampling adapter: wraps MCP sampling with concurrency control."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any


class EmptyModelResponse(RuntimeError):
    """Raised when the model returns an empty response after one retry."""


class SamplingAdapter:
    """Wraps MCP context sampling with a semaphore for concurrency control.

    The semaphore gates how many simultaneous LLM calls are in flight.
    With ``max_concurrent=1`` (the default), parallel callers serialize
    through the LLM one at a time.
    """

    def __init__(self, mcp_context: Any, max_concurrent: int = 1) -> None:
        self._mcp_context = mcp_context
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        """Acquire the semaphore and perform a sampling call.

        Returns the raw text response. If the model returns an empty or
        whitespace-only response, retries once. If the retry is also empty,
        raises ``EmptyModelResponse``.
        """
        async with self._semaphore:
            text = await self._do_sample(
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
            )
            if not text or not text.strip():
                text = await self._do_sample(
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                )
                if not text or not text.strip():
                    raise EmptyModelResponse(
                        "Model returned an empty response after one retry."
                    )
            return text

    async def complete_structured(
        self,
        system_prompt: str,
        messages: list[dict],
        schema: dict,
        temperature: float = 0.3,
    ) -> dict:
        """Perform a sampling call and parse the response as JSON.

        Enhances the system prompt to instruct the LLM to return JSON
        conforming to the provided schema, then parses the response.
        """
        schema_str = json.dumps(schema, indent=2)
        enhanced_system = (
            f"{system_prompt}\n\n"
            "You MUST respond with valid JSON only — no prose, no markdown commentary.\n"
            f"The JSON must conform to this schema:\n{schema_str}\n"
            "Output ONLY the JSON object, optionally wrapped in ```json ... ``` fences."
        )
        text = await self.complete(
            system_prompt=enhanced_system,
            messages=messages,
            temperature=temperature,
        )
        return self._parse_json(text)

    # ------------------------------------------------------------------
    # Overridable sampling call
    # ------------------------------------------------------------------

    async def _do_sample(self, **kwargs: Any) -> str:
        """Perform the actual MCP sampling call.

        Override in tests or subclasses. Raises RuntimeError if no
        mcp_context is available.
        """
        if self._mcp_context is None:
            raise RuntimeError(
                "mcp_context is not set. Provide an MCP context at construction time."
            )
        # Delegate to the MCP context's sampling API
        return await self._mcp_context.sample(**kwargs)

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from a response string.

        Tries direct json.loads first; if that fails, extracts content
        from markdown code fences (```json ... ```).

        Raises:
            ValueError: if neither strategy succeeds.
        """
        # Strategy 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract from ```json ... ``` fences
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if match:
            candidate = match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {text[:200]!r}")
