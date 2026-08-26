"""Download and validate the FinBERT2 checkpoints used by this project."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


MODELS = {
    "finbert2_base": ("valuesimplex-ai-lab/FinBERT2-base", "finbert2-base"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=20.0)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    requested = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(requested).difference(MODELS)
    if unknown:
        raise ValueError(f"unknown FinBERT2 models: {sorted(unknown)}")

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    args.model_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    for alias in requested:
        repo_id, directory_name = MODELS[alias]
        target = args.model_root / directory_name
        marker = target / ".finbert2_complete.json"
        if marker.is_file():
            config = AutoConfig.from_pretrained(target, local_files_only=True)
            AutoTokenizer.from_pretrained(target, local_files_only=True, use_fast=True)
            completed.append({
                "alias": alias,
                "repo_id": repo_id,
                "path": str(target),
                "hidden_size": int(config.hidden_size),
                "cached": True,
            })
            continue

        staging = args.model_root / f".{directory_name}.download"
        for attempt in range(args.retries + 1):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=staging,
                    ignore_patterns=("*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot"),
                    max_workers=args.max_workers,
                )
                break
            except Exception as exc:
                if attempt >= args.retries:
                    raise
                print(json.dumps({
                    "status": "retrying",
                    "model": alias,
                    "attempt": attempt + 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }, ensure_ascii=False), flush=True)
                time.sleep(args.retry_delay)

        config = AutoConfig.from_pretrained(staging, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(staging, local_files_only=True, use_fast=True)
        model = AutoModel.from_pretrained(
            staging, local_files_only=True, add_pooling_layer=False
        )
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        if int(config.max_position_embeddings) < 512:
            raise ValueError("FinBERT2 checkpoint cannot encode the paper's 512-token input")
        if tokenizer.pad_token_id is None:
            raise ValueError("FinBERT2 tokenizer has no padding token")
        del model
        if target.exists():
            raise FileExistsError(f"refusing to replace incomplete model directory: {target}")
        os.replace(staging, target)
        payload = {
            "alias": alias,
            "repo_id": repo_id,
            "hidden_size": int(config.hidden_size),
            "max_position_embeddings": int(config.max_position_embeddings),
            "parameter_count": parameter_count,
        }
        marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        completed.append({**payload, "path": str(target), "cached": False})
    print(json.dumps({"models": completed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
