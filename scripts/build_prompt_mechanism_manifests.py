"""Build the eight cache tasks and 288 rolling mechanism fold tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("roberta", "bge_m3", "ckip_bert", "xlm_roberta_large")
LENGTHS = ("short", "long")
TARGETS = ("event_return_3d", "next_day_return")
TEST_YEARS = tuple(range(2018, 2027))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    args = parser.parse_args()
    cache_rows = [
        {"task_id": index, "model": model, "prompt_length": length}
        for index, (model, length) in enumerate(
            (model, length) for model in MODELS for length in LENGTHS
        )
    ]
    fold_rows = []
    for model in MODELS:
        for length in LENGTHS:
            for variant in (length, f"masked_{length}"):
                for target in TARGETS:
                    for year in TEST_YEARS:
                        fold_rows.append({
                            "task_id": len(fold_rows), "model": model,
                            "prompt_length": length, "variant": variant,
                            "target": target, "test_year": year,
                        })
    _write(args.cache_manifest, cache_rows)
    _write(args.fold_manifest, fold_rows)
    summary = {
        "format_version": "prompt_mechanism_manifests_v1",
        "cache_tasks": len(cache_rows), "fold_tasks": len(fold_rows),
        "models": list(MODELS), "prompt_lengths": list(LENGTHS),
        "targets": list(TARGETS), "test_years": list(TEST_YEARS),
    }
    args.fold_manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
