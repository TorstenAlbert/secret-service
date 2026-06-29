"""Adapter factory: choose the LLM provider and bind per-agent models.

Selection priority:
  1. OpenRouter, if OPENROUTER_API_KEY is set (preferred).
  2. MCP client sampling, if a live MCP context is available (fallback).
  3. Disabled — no provider; sampling calls raise and solve() fails fast.

NOTE: most MCP clients (Claude Code included) do not implement the sampling
capability, and MCP sampling is deprecated as of spec 2026-07-28. OpenRouter is
the recommended provider; the sampling path exists for clients that support it.
"""
from __future__ import annotations

import logging
from typing import Any

from ss.config import Config
from ss.sampling.adapter import SamplingAdapter
from ss.sampling.openrouter import OpenRouterAdapter

logger = logging.getLogger(__name__)

_DISABLED_MSG = (
    "No LLM provider available: set OPENROUTER_API_KEY, or use an MCP client "
    "that supports sampling. (Claude Code does not implement sampling.)"
)


class AdapterBundle:
    """Selects OpenRouter (preferred) or MCP sampling (fallback) per agent.

    With neither a key nor an MCP context, the bundle is disabled and any
    sampling call raises. ``bind_context`` lets the server attach a live MCP
    context per request to enable the sampling fallback.
    """

    def __init__(self, config: Config, mcp_context: Any = None) -> None:
        self._config = config
        self._mcp_context = mcp_context
        self._use_openrouter = bool(config.openrouter_api_key)
        self._cache: dict[str, SamplingAdapter] = {}
        self._mcp_adapter: SamplingAdapter | None = None
        if not self._use_openrouter and mcp_context is not None:
            self._mcp_adapter = SamplingAdapter(
                mcp_context, max_concurrent=config.max_concurrent_sampling
            )

    @property
    def uses_openrouter(self) -> bool:
        """True iff an OpenRouter key is configured (the preferred provider)."""
        return self._use_openrouter

    @property
    def sampling_available(self) -> bool:
        """True iff an MCP context is present to attempt client sampling."""
        return self._mcp_adapter is not None

    @property
    def llm_available(self) -> bool:
        """True iff some LLM provider (OpenRouter or sampling) can be used."""
        return self._use_openrouter or self.sampling_available

    def bind_context(self, mcp_context: Any) -> None:
        """Attach a live MCP context for the sampling fallback.

        No-op in OpenRouter mode (OpenRouter is always preferred) or when the
        context is None. Otherwise (re)creates the lightweight MCP adapter.
        """
        if self._use_openrouter or mcp_context is None:
            return
        self._mcp_context = mcp_context
        self._mcp_adapter = SamplingAdapter(
            mcp_context, max_concurrent=self._config.max_concurrent_sampling
        )

    def _openrouter_for(self, model: str) -> OpenRouterAdapter:
        return OpenRouterAdapter(
            api_key=self._config.openrouter_api_key,  # type: ignore[arg-type]
            model=model,
            base_url=self._config.openrouter_base_url,
            timeout=self._config.openrouter_timeout,
            site_url=self._config.openrouter_site_url,
            site_name=self._config.openrouter_site_name,
            max_concurrent=self._config.max_concurrent_sampling,
        )

    def for_agent(self, agent_name: str) -> SamplingAdapter:
        """Return the adapter for a pipeline agent (OpenRouter preferred, else sampling)."""
        if self._use_openrouter:
            if agent_name not in self._cache:
                self._cache[agent_name] = self._openrouter_for(
                    self._config.model_for_agent(agent_name)
                )
            return self._cache[agent_name]
        if self._mcp_adapter is not None:
            return self._mcp_adapter
        raise RuntimeError(_DISABLED_MSG)

    def council_member_models(self) -> list[str]:
        """Model ids for the council members."""
        return list(self._config.openrouter_agent_models.council_members)

    def council_chairman_model(self) -> str:
        """Model id for the council chairman."""
        return self._config.openrouter_agent_models.council_chairman

    def _council_adapter(self) -> OpenRouterAdapter:
        if "__council__" not in self._cache:
            self._cache["__council__"] = self._openrouter_for(
                self._config.openrouter_default_model
            )
        return self._cache["__council__"]  # type: ignore[return-value]

    async def council_complete(
        self,
        *,
        model: str | None,
        system_prompt: str,
        messages: list[dict],
        temperature: float,
    ) -> str:
        """Council call: OpenRouter per-model; else MCP sampling; else raise."""
        if self._use_openrouter:
            return await self._council_adapter().sample(
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
                model=model,
            )
        if self._mcp_adapter is not None:
            return await self._mcp_adapter.complete(
                system_prompt=system_prompt, messages=messages, temperature=temperature
            )
        raise RuntimeError(_DISABLED_MSG)


def build_adapter(config: Config, mcp_context: Any = None) -> AdapterBundle:
    """Construct an AdapterBundle: OpenRouter preferred, MCP sampling fallback, else disabled."""
    bundle = AdapterBundle(config, mcp_context)
    if bundle.uses_openrouter:
        logger.info("LLM provider: OpenRouter (default model=%s)", config.openrouter_default_model)
    elif bundle.sampling_available:
        logger.info("LLM provider: MCP client sampling (no OpenRouter key set)")
    else:
        logger.warning(
            "No LLM provider configured (no OPENROUTER_API_KEY, no MCP sampling context). "
            "solve() will return a configuration error; read-only tools still work."
        )
    return bundle
