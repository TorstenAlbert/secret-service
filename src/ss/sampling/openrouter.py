"""OpenRouter LLM provider adapter (OpenAI-compatible chat completions)."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from ss.sampling.adapter import SamplingAdapter

logger = logging.getLogger(__name__)


class OpenRouterAdapter(SamplingAdapter):
    """SamplingAdapter backed by OpenRouter's OpenAI-compatible chat API.

    Reuses the parent's semaphore and JSON parsing; overrides only the actual
    network call. Each instance binds a default model; individual calls may
    override it via the ``model`` keyword (used by the multi-model Council).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 120.0,
        site_url: str = "",
        site_name: str = "secret-service",
        max_concurrent: int = 1,
    ) -> None:
        super().__init__(mcp_context=None, max_concurrent=max_concurrent)
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._site_url = site_url
        self._site_name = site_name
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        """Build request headers, including auth and optional attribution."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._site_name:
            headers["X-Title"] = self._site_name
        return headers

    async def sample(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        model: str | None = None,
    ) -> str:
        """Semaphore-guarded sample with an optional per-call model override."""
        async with self._semaphore:
            return await self._do_sample(
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
                model=model,
            )

    async def _do_sample(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        payload = {
            "model": model or self._model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
        }
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter returned status {response.status_code}: {response.text[:400]}"
            )

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Malformed OpenRouter response: {str(response.text)[:400]}"
            ) from exc
