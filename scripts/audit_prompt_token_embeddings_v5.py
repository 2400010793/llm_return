"""Validate v5 prompt-token shard coverage, shapes, IDs, and row alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("data/processed/prompt_token_embeddings_v5")
COMBINATIONS = (
    ("roberta", "short", 28, 768), ("roberta", "masked_short", 28, 768),
    ("bge_m3", "short", 18, 1024), ("bge_m3", "masked_short", 18, 1024),
    ("roberta", "long", 180, 768), ("roberta", "masked_long", 180, 768),
    ("bge_m3", "long", 110, 1024), ("bge_m3", "masked_long", 110, 1024),
)
EXPECTED_ROWS = 350577


def main() -> None:
    results = []
    for model, variant, token_count, hidden_size in COMBINATIONS:
        directories = sorted(ROOT.glob(f"shard-*/{model}/{variant}"))
        rows = []
        errors = []
        for directory in directories:
            try:
                summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                matrix = np.load(directory / "prompt_token_embeddings.npy", mmap_mode="r")
                ids = np.load(directory / "prompt_input_ids.npy", mmap_mode="r")
                expected_shape = (int(summary["rows"]), token_count, hidden_size)
                if matrix.shape != expected_shape:
                    errors.append(f"{directory}: shape {matrix.shape} != {expected_shape}")
                if ids.shape != (token_count,):
                    errors.append(f"{directory}: prompt IDs shape {ids.shape}")
                if summary.get("prompt_slice") != f"[1:{1 + token_count}] after BOS":
                    errors.append(f"{directory}: incorrect prompt slice")
                with (directory / "metadata.jsonl").open(encoding="utf-8") as handle:
                    local = [int(json.loads(line)["row_index"]) for line in handle if line.strip()]
                if len(local) != expected_shape[0]:
                    errors.append(f"{directory}: metadata rows {len(local)} != {expected_shape[0]}")
                rows.extend(local)
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{directory}: {exc}")
        complete = (
            len(directories) == 32 and len(rows) == EXPECTED_ROWS
            and len(set(rows)) == EXPECTED_ROWS
            and min(rows, default=0) == 1 and max(rows, default=0) == EXPECTED_ROWS
            and not errors
        )
        results.append({
            "model": model, "variant": variant, "shards": len(directories),
            "rows": len(rows), "unique_rows": len(set(rows)),
            "token_count": token_count, "hidden_size": hidden_size,
            "complete": complete, "errors": errors,
        })
    report = {"all_complete": all(item["complete"] for item in results), "results": results}
    output = Path("reports/prompt_token_embeddings_v5_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()