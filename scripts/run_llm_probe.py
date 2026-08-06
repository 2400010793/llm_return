"""Small, explicit probe for local Ollama or OpenAI-compatible chat models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.api_embeddings import api_chat_completion
from src.text.ollama_client import ollama_chat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--provider", choices=("ollama", "openai-compatible"), default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()
    if args.provider == "ollama":
        text = ollama_chat(args.prompt, model=args.model or "qwen2.5:32b", base_url=args.base_url or "http://127.0.0.1:11434")
    else:
        text = api_chat_completion(args.prompt, model=args.model or "gpt-4o-mini", base_url=args.base_url or "https://api.openai.com/v1")
    print(json.dumps({"provider": args.provider, "model": args.model, "response": text}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()