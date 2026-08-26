"""Build auditable Qwen and same-panel linear baseline manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    "task_id", "model", "embedding_root", "variant", "target", "regressor",
    "run_mode", "reducer", "components", "seed", "output",
)


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for task_id, row in enumerate(rows):
            writer.writerow({"task_id": task_id, **row})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    target = "next_day_open_to_open_winsor_residual"
    rows: list[dict[str, object]] = []
    models = (
        ("qwen3_embedding_8b", "data/processed/pooled_embeddings_qwen_candidates_1000", ("plain", "short", "masked_short")),
        ("roberta", "data/processed/pooled_embeddings_v3", ("short", "masked_short")),
        ("bge_m3", "data/processed/pooled_embeddings_v3", ("short", "masked_short")),
    )
    configurations = [
        *(('ridge', 'none', 0),),
        *(("ridge", "pca", value) for value in (32, 64, 128, 256)),
        *(("huber_sgd", "pca", value) for value in (32, 64, 128, 256)),
    ]
    for model, root, variants in models:
        for variant in variants:
            for regressor, reducer, components in configurations:
                stem = f"{model}_{variant}_{regressor}_{reducer}{components or ''}_seed42"
                rows.append({
                    "model": model, "embedding_root": root, "variant": variant,
                    "target": target, "regressor": regressor, "run_mode": "screen",
                    "reducer": reducer, "components": components or 128, "seed": 42,
                    "output": str(args.result_root / "screen" / f"{stem}.json"),
                })
    write(args.output, rows)
    print(f"wrote {len(rows)} tasks to {args.output}")


if __name__ == "__main__":
    main()
