"""Audit the six neutral prompt token budgets for one local tokenizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from scripts.run_short_pooled_embeddings import MODEL_PATHS
from src.data.aligned_prompts import _prompt_audit, tokenizer_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prompts = config.get("prompts", {})
    if config.get("variant") != "masked_short" or len(prompts) != 6:
        raise ValueError("config must contain six masked_short prompts")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS[args.model], local_files_only=True, use_fast=True,
    )
    audits = {}
    for prompt_id, item in prompts.items():
        text = str(item["text"])
        if any(marker in text for marker in ("高", "低", "中")):
            raise ValueError(f"neutral prompt contains a level marker: {text}")
        audits[prompt_id] = _prompt_audit(text, str(item["target"]), tokenizer)
    report = {
        "format_version": "neutral_masked_short_preflight_v1",
        "model": args.model,
        "model_path": MODEL_PATHS[args.model],
        "tokenizer_sha256": tokenizer_digest(tokenizer),
        "prompts": audits,
        "cross_prompt_token_counts_may_differ": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
