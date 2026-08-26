"""Build bounded segment-gating experiments from existing pooled embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_pooled_classification_manifest import ROOTS, completed_rows
from src.data.pooled_embeddings import discover_pooled_parts


MODELS = ("roberta", "bge_m3")
VARIANTS = ("short", "masked_short")
GATES = ("variance", "logistic_l1")
KEEP_GROUPS = (2,)
CLASSIFIERS = ("simple_mlp",)


def build_rows(root: Path, expected_rows: int = 350577) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for variant in VARIANTS:
            embedding_root = ROOTS[variant]
            parts = discover_pooled_parts(root / embedding_root, model, variant)
            if len(parts) != 32 or completed_rows(parts) != expected_rows:
                raise ValueError(
                    f"incomplete pooled assets for {model}/{variant}: "
                    f"shards={len(parts)} rows={completed_rows(parts) if parts else 0}"
                )
            for gate_method in GATES:
                for gate_keep in KEEP_GROUPS:
                    for classifier in CLASSIFIERS:
                        stem = (
                            f"{model}_{variant}_title_body_full_concat_"
                            f"{gate_method}_keep{gate_keep}_{classifier}"
                        )
                        rows.append({
                            "task_id": len(rows),
                            "embedding_root": str(embedding_root),
                            "model": model,
                            "variant": variant,
                            "feature": "title_body_full_concat",
                            "classifier": classifier,
                            "gate_method": gate_method,
                            "gate_keep": gate_keep,
                            "output": (
                                "reports/classification/pooled_embeddings/next1_segment_gating/"
                                f"{stem}.json"
                            ),
                            "train_target": "next_day_return",
                            "evaluation_target": "next_day_return",
                        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-rows", type=int, default=350577)
    parser.add_argument(
        "--output", type=Path,
        default=Path("configs/generated/pooled_segment_gating_seed42.tsv"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = build_rows(root, args.expected_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "tasks": len(rows),
        "models": list(MODELS),
        "variants": list(VARIANTS),
        "feature": "title_body_full_concat",
        "gate_methods": list(GATES),
        "keep_groups": list(KEEP_GROUPS),
        "classifiers": list(CLASSIFIERS),
        "generates_embeddings": False,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
