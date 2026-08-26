"""Build matrix-assembly tasks for no-neutral-marker anchor embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--representation", default="body_mean")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    for task_id, (axis, item) in enumerate(config["prompts"].items()):
        rows.append({
            "task_id": task_id,
            "kind": "anchor_matrix",
            "axis": axis,
            "level": "anchor_without_neutral_marker",
            "prompt_id": item["prompt_id"],
            "model": "roberta",
            "variant": "masked_short",
            "representation": args.representation,
            "shards": args.shards,
            "embeddings_root": str(args.embeddings_root),
            "output_root": str(args.output_root),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(json.dumps({"tasks": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
