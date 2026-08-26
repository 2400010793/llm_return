"""Build a manifest for ranking every position in frozen v5 prompt tokens."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("roberta", "bge_m3")
VARIANTS = ("short", "masked_short")
TARGETS = ("next_day_return", "event_return_3d")
METHODS = ("fisher", "variance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("data/processed/prompt_token_embeddings_v5"),
    )
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--keep-tokens", type=int, default=8)
    parser.add_argument(
        "--output", type=Path,
        default=Path("configs/generated/prompt_token_ranking_v5.tsv"),
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for variant in VARIANTS:
            shards = sorted(args.input_root.glob(f"shard-*/{model}/{variant}"))
            complete = [
                shard for shard in shards
                if all((shard / name).is_file() for name in (
                    "metadata.jsonl", "prompt_token_embeddings.npy", "prompt_tokens.json",
                ))
            ]
            if len(complete) != args.expected_shards:
                raise ValueError(
                    f"incomplete prompt tokens for {model}/{variant}: "
                    f"{len(complete)}/{args.expected_shards}"
                )
            for target in TARGETS:
                for method in METHODS:
                    output = Path("reports/prompt_token_rankings_v5") / (
                        f"{model}_{variant}_{target}_{method}_keep{args.keep_tokens}.json"
                    )
                    rows.append({
                        "task_id": len(rows), "model": model, "variant": variant,
                        "target": target, "method": method,
                        "keep_tokens": args.keep_tokens, "output": str(output),
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "tasks": len(rows), "models": list(MODELS), "variants": list(VARIANTS),
        "targets": list(TARGETS), "methods": list(METHODS),
        "keep_tokens": args.keep_tokens, "generates_embeddings": False,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
