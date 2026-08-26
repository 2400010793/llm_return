"""Probe downloadable FinBERT checkpoints with classification-only inference."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODELS = (
    "yiyanghkust/finbert-tone-chinese",
    "hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2",
    "ProsusAI/finbert",
)


def snapshot_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def json_labels(config: Any) -> dict[str, str]:
    values = getattr(config, "id2label", None) or {}
    return {str(key): str(value) for key, value in values.items()}


def probe_model(model_id: str, cache_dir: str, max_length: int) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "model": model_id,
        "status": "failed",
        "embedding_generated": False,
    }
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_files_only=False,
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            snapshot, local_files_only=True
        )
        model.eval()
        texts = [
            "公司公布业绩增长，市场关注后续盈利能力。",
            "公司公告预计亏损，经营风险明显上升。",
        ]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
        result.update(
            {
                "status": "deployed",
                "snapshot": str(snapshot),
                "snapshot_bytes": snapshot_bytes(snapshot),
                "num_parameters": int(sum(p.numel() for p in model.parameters())),
                "num_labels": int(model.config.num_labels),
                "id2label": json_labels(model.config),
                "tokenizer_class": tokenizer.__class__.__name__,
                "model_class": model.__class__.__name__,
                "probe_rows": len(texts),
                "probability_shape": list(probabilities.shape),
                "probabilities_finite": bool(torch.isfinite(probabilities).all()),
                "predicted_label_ids": probabilities.argmax(dim=-1).tolist(),
                "max_probability": probabilities.max(dim=-1).values.tolist(),
            }
        )
    except Exception as exc:  # Record one unavailable checkpoint without hiding others.
        result.update(
            {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            }
        )
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated Hugging Face model IDs",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")),
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/finbert/finbert_deployment_probe.json"),
    )
    args = parser.parse_args()
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    if not models:
        raise ValueError("at least one model ID is required")

    results = [probe_model(model_id, args.cache_dir, args.max_length) for model_id in models]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Download and run classification-only smoke tests; no embeddings generated.",
        "cache_dir": args.cache_dir,
        "max_length": args.max_length,
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
