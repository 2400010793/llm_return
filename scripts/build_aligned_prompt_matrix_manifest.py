"""Build shard-assembly tasks for the aligned masked-short prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.aligned_prompts import LEVELS, load_aligned_prompt_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--representation", default="body_mean")
    args = parser.parse_args()
    config = load_aligned_prompt_config(args.config)
    rows = []
    task_id = 0
    for axis, spec in config["axes"].items():
        for level in LEVELS:
            rows.append({
                "task_id": task_id,
                "kind": "matrix",
                "axis": axis,
                "level": level,
                "prompt_id": spec["levels"][level]["prompt_id"],
                "model": "roberta",
                "variant": "masked_short",
                "representation": args.representation,
                "shards": args.shards,
                "embeddings_root": str(args.embeddings_root),
                "output_root": str(args.output_root),
            })
            task_id += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(json.dumps({"tasks": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
