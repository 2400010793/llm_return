"""Audit full shard/row coverage without loading large embedding matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COMBINATIONS = (
    (Path("data/processed/pooled_embeddings_v3"), "roberta", "short"),
    (Path("data/processed/pooled_embeddings_v3"), "roberta", "masked_short"),
    (Path("data/processed/pooled_embeddings_v3"), "bge_m3", "short"),
    (Path("data/processed/pooled_embeddings_v3"), "bge_m3", "masked_short"),
    (Path("data/processed/pooled_long_embeddings_v4"), "roberta", "long"),
    (Path("data/processed/pooled_long_embeddings_v4"), "roberta", "masked_long"),
    (Path("data/processed/pooled_long_embeddings_v4"), "bge_m3", "long"),
    (Path("data/processed/pooled_long_embeddings_v4"), "bge_m3", "masked_long"),
)
REQUIRED_OUTPUTS = (
    "prompt_token_embeddings", "prompt_mean", "title_mean", "body_mean",
    "title_body_mean", "full_mean",
)


def audit_combination(root: Path, model: str, variant: str, expected_rows: int) -> dict:
    summaries = sorted(root.glob(f"shard-*/{model}/{variant}/summary.json"))
    row_indexes: list[int] = []
    summary_rows = metadata_rows = 0
    dimensions: set[int] = set()
    prompt_slots: set[int] = set()
    errors: list[str] = []
    for summary_path in summaries:
        directory = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = int(summary.get("rows", 0))
        summary_rows += rows
        outputs = summary.get("outputs", {})
        missing = [name for name in REQUIRED_OUTPUTS if name not in outputs]
        if missing:
            errors.append(f"{directory}: missing outputs {missing}")
        for name in ("title_mean", "body_mean", "full_mean"):
            shape = outputs.get(name, [])
            if len(shape) == 2:
                dimensions.add(int(shape[1]))
        prompt_slots.add(int(summary.get("prompt_token_slots", 0)))
        if not (directory / "short_pooling.npz").is_file():
            errors.append(f"{directory}: missing short_pooling.npz")
        metadata_path = directory / "metadata.jsonl"
        if not metadata_path.is_file():
            errors.append(f"{directory}: missing metadata.jsonl")
            continue
        local_rows = 0
        with metadata_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                local_rows += 1
                row_indexes.append(int(json.loads(line)["row_index"]))
        metadata_rows += local_rows
        if local_rows != rows:
            errors.append(f"{directory}: metadata rows {local_rows} != summary rows {rows}")
    unique_rows = len(set(row_indexes))
    complete = (
        len(summaries) == 32
        and summary_rows == expected_rows
        and metadata_rows == expected_rows
        and unique_rows == expected_rows
        and min(row_indexes, default=0) == 1
        and max(row_indexes, default=0) == expected_rows
        and len(dimensions) == 1
        and not errors
    )
    return {
        "root": str(root), "model": model, "variant": variant,
        "shards": len(summaries), "summary_rows": summary_rows,
        "metadata_rows": metadata_rows, "unique_row_indexes": unique_rows,
        "row_index_min": min(row_indexes, default=None),
        "row_index_max": max(row_indexes, default=None),
        "embedding_dimensions": sorted(dimensions),
        "prompt_token_slots": sorted(prompt_slots),
        "pooled_outputs_complete": complete,
        "prompt_token_embeddings_final_valid": False,
        "prompt_token_note": "Known BOS offset: current prompt token slice includes BOS and omits the final prompt token.",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-rows", type=int, default=350577)
    parser.add_argument("--output", type=Path, default=Path("reports/embedding_completion_audit.json"))
    args = parser.parse_args()
    combinations = [audit_combination(*combo, args.expected_rows) for combo in COMBINATIONS]
    report = {
        "expected_rows": args.expected_rows,
        "all_pooled_combinations_complete": all(row["pooled_outputs_complete"] for row in combinations),
        "all_prompt_token_outputs_final_valid": False,
        "combinations": combinations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_pooled_combinations_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()