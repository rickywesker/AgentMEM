"""Minimal async client for OpenAI-compatible chat and embedding endpoints.

Deliberately thin: one POST, bounded retries, no SDK. Every provider we target
(DeepSeek direct, DashScope compatible-mode, the team proxy) speaks this shape.
"""

from __future__ import annotations

import asyncio
import random

import httpx

from .config import Gear

RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


async def chat(
    client: httpx.AsyncClient,
    gear: Gear,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    attempts: int = 6,
) -> str:
    """Single-turn completion. Returns the message content."""
    if not gear.base_url or not gear.model:
        raise LLMError("gear is not configured (missing base_url or model)")

    payload: dict = {
        "model": gear.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.post(
                f"{gear.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {gear.api_key}"},
                json=payload,
            )
            if response.status_code in RETRYABLE:
                raise LLMError(f"retryable {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except (httpx.TransportError, httpx.HTTPStatusError, LLMError, KeyError) as error:
            last = error
            if attempt == attempts - 1:
                break
            await asyncio.sleep(min(2**attempt, 30) * (0.5 + random.random()))
    raise LLMError(f"chat failed after {attempts} attempts: {last}")


async def embed(
    client: httpx.AsyncClient,
    gear: Gear,
    texts: list[str],
    *,
    attempts: int = 6,
) -> list[list[float]]:
    """Batch embedding. Order of the returned vectors matches ``texts``."""
    if not texts:
        return []
    if not gear.base_url or not gear.model:
        raise LLMError("embed gear is not configured")

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.post(
                f"{gear.base_url}/embeddings",
                headers={"Authorization": f"Bearer {gear.api_key}"},
                json={"model": gear.model, "input": texts},
            )
            if response.status_code in RETRYABLE:
                raise LLMError(f"retryable {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            rows = response.json()["data"]
            rows.sort(key=lambda row: row.get("index", 0))
            return [row["embedding"] for row in rows]
        except (httpx.TransportError, httpx.HTTPStatusError, LLMError, KeyError) as error:
            last = error
            if attempt == attempts - 1:
                break
            await asyncio.sleep(min(2**attempt, 30) * (0.5 + random.random()))
    raise LLMError(f"embed failed after {attempts} attempts: {last}")
