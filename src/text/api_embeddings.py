"""OpenAI-compatible embedding client with environment-only API keys.

No key is read from project files or command-line arguments. Set the provider's
key in the process environment before running an experiment.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import numpy as np


def api_chat_completion(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    system: str = "You are a careful financial text analysis assistant.",
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> str:
    """Call an OpenAI-compatible chat model using an environment-only key."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment; never put an API key in source or config files.")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"chat API returned HTTP {exc.code}: {body}") from exc
    except OSError as exc:
        raise RuntimeError(f"chat API request failed: {exc}") from exc
    try:
        return str(result["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("chat API returned an invalid response") from exc


def encode_api_embeddings(
    texts: list[str],
    *,
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    batch_size: int = 64,
    timeout: float = 120.0,
) -> np.ndarray:
    """Call an OpenAI-compatible ``/embeddings`` endpoint in batches.

    The key comes from ``api_key`` only for library callers, otherwise from
    ``OPENAI_API_KEY``. Use a compatible endpoint and key for other providers.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment; never put an API key in source or config files.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        payload = json.dumps({"input": texts[start : start + batch_size], "model": model}).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"embedding API returned HTTP {exc.code}: {body}") from exc
        except OSError as exc:
            raise RuntimeError(f"embedding API request failed: {exc}") from exc
        data = result.get("data")
        if not isinstance(data, list) or len(data) != min(batch_size, len(texts) - start):
            raise RuntimeError("embedding API returned an invalid data length")
        vectors.extend(item["embedding"] for item in sorted(data, key=lambda item: item.get("index", 0)))
    return np.asarray(vectors, dtype=np.float32)
