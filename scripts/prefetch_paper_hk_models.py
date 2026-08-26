"""Download and validate the two China (HK) encoders used by CKX.

Downloads are separated from embedding arrays and promoted atomically after a
local-only Transformers validation.  Interrupted downloads stay in a staging
directory and can be resumed without exposing an incomplete model directory.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


MODELS = {
    "ckip_bert": ("ckiplab/bert-base-chinese", "ckiplab-bert-base-chinese"),
    "xlm_roberta_large": ("xlm-roberta-large", "xlm-roberta-large"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument(
        "--retries", type=int, default=0,
        help="Retry each interrupted snapshot this many times; completed chunks are reused.",
    )
    parser.add_argument("--retry-delay", type=float, default=20.0)
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Keep direct downloads conservative and reduce parallel connection failures.",
    )
    args = parser.parse_args()
    if args.retries < 0 or args.retry_delay < 0 or args.max_workers < 1:
        raise ValueError("retries/delay/workers must be non-negative/non-negative/positive")

    requested = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(requested).difference(MODELS)
    if unknown:
        raise ValueError(f"unknown paper models: {sorted(unknown)}")

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    args.model_root.mkdir(parents=True, exist_ok=True)
    completed = []
    for alias in requested:
        repo_id, directory_name = MODELS[alias]
        target = args.model_root / directory_name
        marker = target / ".paper_hk_complete.json"
        if marker.is_file():
            AutoConfig.from_pretrained(target, local_files_only=True)
            AutoTokenizer.from_pretrained(target, local_files_only=True, use_fast=True)
            completed.append({"alias": alias, "repo_id": repo_id, "path": str(target), "cached": True})
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
                    "alias": alias,
                    "repo_id": repo_id,
                    "attempt": attempt + 1,
                    "remaining_retries": args.retries - attempt,
                    "delay_seconds": args.retry_delay,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }, ensure_ascii=False), flush=True)
                time.sleep(args.retry_delay)
        AutoConfig.from_pretrained(staging, local_files_only=True)
        AutoTokenizer.from_pretrained(staging, local_files_only=True, use_fast=True)
        if target.exists():
            raise FileExistsError(
                f"refusing to replace unvalidated existing model directory: {target}"
            )
        os.replace(staging, target)
        marker.write_text(
            json.dumps({"alias": alias, "repo_id": repo_id}, indent=2) + "\n",
            encoding="utf-8",
        )
        completed.append({"alias": alias, "repo_id": repo_id, "path": str(target), "cached": False})
    print(json.dumps({"models": completed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
