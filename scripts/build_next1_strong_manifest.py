"""Build bounded strong-classifier manifests for the direct next-day task."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOTS = {
    "short": Path("data/processed/pooled_embeddings_v3"),
    "masked_short": Path("data/processed/pooled_embeddings_v3"),
}

# Validation-selected, masking ablation, cross-encoder replication, and
# segment-preserving input.  This is intentionally not a full Cartesian sweep.
SCREEN_REPRESENTATIONS = (
    ("roberta", "short", "body_mean"),
    ("roberta", "masked_short", "body_mean"),
    ("bge_m3", "masked_short", "body_mean"),
    ("roberta", "masked_short", "title_body_full_concat"),
)
SCREEN_CLASSIFIERS = (
    "hist_gradient_boosting", "extra_trees", "xgboost", "lightgbm", "catboost",
)


def build_rows(
    root: Path,
    *,
    phase: str,
    include_existing: bool = False,
    classifiers: tuple[str, ...] = SCREEN_CLASSIFIERS,
    seeds: tuple[int, ...] = (42,),
) -> list[dict[str, object]]:
    if phase not in {"screen", "fine"}:
        raise ValueError("phase must be screen or fine")
    unknown = set(classifiers).difference(SCREEN_CLASSIFIERS)
    if unknown:
        raise ValueError(f"unknown strong classifiers: {sorted(unknown)}")
    if not seeds:
        raise ValueError("at least one seed is required")
    rows: list[dict[str, object]] = []
    for model, variant, feature in SCREEN_REPRESENTATIONS:
        embedding_root = ROOTS[variant]
        if not (root / embedding_root).is_dir():
            continue
        for classifier in classifiers:
            for seed in seeds:
                stem = f"{model}_{variant}_{feature}_{classifier}_none_0_seed{seed}"
                output = Path(
                    f"reports/classification/pooled_embeddings/next1_strong_{phase}/{stem}.json"
                )
                if not include_existing and (root / output).is_file():
                    continue
                rows.append({
                    "task_id": len(rows), "phase": f"next1_strong_{phase}",
                    "embedding_root": str(embedding_root), "model": model,
                    "variant": variant, "feature": feature,
                    "classifier": classifier, "reducer": "none", "components": 0,
                    "search_stage": "coarse" if phase == "screen" else "fine",
                    "seed": seed, "output": str(output),
                    "train_target": "next_day_return",
                    "evaluation_target": "next_day_return",
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("screen", "fine"), default="screen")
    parser.add_argument("--classifiers", default=",".join(SCREEN_CLASSIFIERS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    classifiers = tuple(value.strip() for value in args.classifiers.split(",") if value.strip())
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    rows = build_rows(
        args.root, phase=args.phase, include_existing=args.include_existing,
        classifiers=classifiers, seeds=seeds,
    )
    if not rows:
        raise ValueError("no next1 strong-model rows were generated")
    output = args.output or Path(f"configs/generated/pooled_next1_strong_{args.phase}.tsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "phase": args.phase, "tasks": len(rows), "output": str(output),
        "classifiers": sorted({str(row["classifier"]) for row in rows}),
        "representations": sorted({
            f'{row["model"]}:{row["variant"]}:{row["feature"]}' for row in rows
        }),
        "seeds": sorted({int(row["seed"]) for row in rows}),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
