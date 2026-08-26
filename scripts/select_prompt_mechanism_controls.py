"""Select leakage-safe long-prompt encoder controls from 2026 validation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def _placebo_positions(generic: list[int], count: int) -> list[int]:
    if count > len(generic):
        raise ValueError("generic prompt span is too small for an equal-length placebo")
    if count == 1:
        return [generic[len(generic) // 2]]
    indexes = [round(index * (len(generic) - 1) / (count - 1)) for index in range(count)]
    return sorted(generic[index] for index in indexes)


def select_controls(summary_root: Path, cache_root: Path) -> dict[str, object]:
    audit = json.loads((summary_root / "audit.json").read_text(encoding="utf-8"))
    if int(audit.get("completed_folds", 0)) != int(audit.get("expected_full_folds", -1)):
        raise ValueError("control selection requires all 288 first-stage folds")
    yearly = pd.read_parquet(summary_root / "classification_by_year.parquet")
    candidates = yearly[
        yearly["prompt_length"].eq("long")
        & yearly["test_year"].eq(2026)
        & yearly["classifier"].eq("logistic")
        & yearly["representation"].eq("full_prompt_mean")
    ].copy()
    if candidates.empty:
        raise ValueError("no 2026 long-prompt validation candidates")
    selections = []
    controls: dict[tuple[str, str], set[str]] = {}
    for target, frame in candidates.groupby("target", sort=True):
        winner = frame.sort_values(
            ["validation_accuracy", "model", "variant"],
            ascending=[False, True, True], kind="stable",
        ).iloc[0]
        model, variant = str(winner["model"]), str(winner["variant"])
        baseline_accuracy = float(winner["validation_accuracy"])
        leaveout = yearly[
            yearly["model"].eq(model) & yearly["variant"].eq(variant)
            & yearly["target"].eq(target) & yearly["test_year"].eq(2026)
            & yearly["classifier"].eq("logistic")
            & yearly["representation"].astype(str).str.startswith("leaveout::")
        ].copy()
        leaveout["semantic_group"] = leaveout["representation"].str.split("::").str[1]
        leaveout["validation_contribution"] = baseline_accuracy - leaveout["validation_accuracy"]
        non_direction = leaveout[
            ~leaveout["semantic_group"].isin(["direction", "generic_instruction"])
        ].sort_values(
            ["validation_contribution", "semantic_group"], ascending=[False, True],
        ).head(2)
        groups = ["direction", *non_direction["semantic_group"].astype(str).tolist()]
        controls.setdefault((model, variant), set()).update(groups)
        selections.append({
            "target": target, "model": model, "variant": variant,
            "validation_accuracy": baseline_accuracy,
            "selected_groups": groups,
            "selection_data": "2026 fold: fit 2018-2023, validation 2024-2025",
        })

    specs = []
    for model_variant, groups in sorted(controls.items()):
        model, variant = model_variant
        cache = cache_root / model / "long" / "manifest.json"
        manifest = json.loads(cache.read_text(encoding="utf-8"))
        generic = list(manifest["group_positions"]["generic_instruction"])
        specs.append({
            "control_id": "no_prompt_position_matched", "model": model,
            "source_variant": variant, "mode": "no_prompt_position_matched",
            "semantic_group": None, "positions": [], "is_placebo": False,
        })
        for group in sorted(groups):
            positions = list(manifest["group_positions"][group])
            specs.extend([
                {
                    "control_id": f"ablate_{group}", "model": model,
                    "source_variant": variant, "mode": "replace_positions",
                    "semantic_group": group, "positions": positions,
                    "is_placebo": False,
                },
                {
                    "control_id": f"placebo_{group}", "model": model,
                    "source_variant": variant, "mode": "replace_positions",
                    "semantic_group": group,
                    "positions": _placebo_positions(generic, len(positions)),
                    "is_placebo": True,
                },
            ])
    return {
        "format_version": "prompt_mechanism_control_selection_v1",
        "selection_scope": "validation_only", "selections": selections,
        "control_specs": specs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--classification-manifest", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=64)
    args = parser.parse_args()
    report = select_controls(args.summary_root, args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    rows = []
    for spec_index, spec in enumerate(report["control_specs"]):
        for shard in range(args.shards):
            rows.append({
                "task_id": len(rows), "spec_index": spec_index, "shard": shard,
                "model": spec["model"], "source_variant": spec["source_variant"],
                "control_id": spec["control_id"],
                "output_variant": f"{spec['source_variant']}__{spec['control_id']}",
            })
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(rows)
    classification_rows = []
    for spec_index, spec in enumerate(report["control_specs"]):
        for target in ("event_return_3d", "next_day_return"):
            classification_rows.append({
                "task_id": len(classification_rows), "spec_index": spec_index,
                "model": spec["model"], "source_variant": spec["source_variant"],
                "control_id": spec["control_id"],
                "output_variant": f"{spec['source_variant']}__{spec['control_id']}",
                "target": target,
            })
    with args.classification_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(classification_rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(classification_rows)
    print(json.dumps({
        "specs": len(report["control_specs"]), "embedding_tasks": len(rows),
        "classification_tasks": len(classification_rows),
        "selection": str(args.output), "manifest": str(args.manifest),
        "classification_manifest": str(args.classification_manifest),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
