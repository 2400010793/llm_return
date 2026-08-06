"""Minimal Ollama HTTP client for optional, reproducible auxiliary signals."""

from __future__ import annotations

import json
import urllib.request
from typing import Any


def ollama_generate(
    prompt: str,
    *,
    model: str = "qwen2.5:32b",
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> str:
    """Call a locally running Ollama model and return its generated text."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except OSError as exc:
        raise RuntimeError(
            "Ollama is not reachable. Start the service separately before calling this client."
        ) from exc
    return str(result.get("response", ""))


def ollama_embed(
    texts: list[str],
    *,
    model: str = "nomic-embed-text",
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 120.0,
) -> list[list[float]]:
    """Create local Ollama embeddings without calling a hidden remote API."""
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except OSError as exc:
        raise RuntimeError(
            "Ollama is not reachable. Start the service separately before calling this client."
        ) from exc
    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list):
        raise RuntimeError("Ollama returned no embeddings")
    return embeddings


def ollama_chat(
    prompt: str,
    *,
    model: str = "qwen2.5:32b",
    base_url: str = "http://127.0.0.1:11434",
    system: str = "You are a careful financial text analysis assistant.",
    temperature: float = 0.0,
    timeout: float = 300.0,
) -> str:
    """Run a local Ollama model, including LLaMA-family models."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except OSError as exc:
        raise RuntimeError("Ollama is not reachable. Start the service separately before calling this client.") from exc
    try:
        return str(result["message"]["content"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Ollama returned an invalid chat response") from exc
