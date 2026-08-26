"""Summarize completed aligned prompt regression bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read(root: Path, variant: str) -> list[dict]:
    rows = []
    for result in sorted(root.glob("*/result.json")):
        payload = json.loads(result.read_text(encoding="utf-8"))
        metrics = payload["results"][0]
        parts = Path(payload["input"]["matrix"]).parts
        if "factors" in parts:
            index = parts.index("factors")
            axis = parts[index + 2]
            factor = parts[-1].removesuffix(".npy")
        else:
            axis, factor = "anchor", "anchor"
        validation = metrics.get("validation_metrics", {})
        rows.append({
            "variant": variant,
            "axis": axis,
            "factor": factor,
            "target": metrics["target"],
            "regressor": metrics["regressor"],
            "target_kind": metrics.get("target_kind", payload.get("design", {}).get("target_kind")),
            "validation_ir": validation.get("rank_ic_information_ratio"),
            "test_ir": metrics.get("rank_ic_information_ratio"),
            "test_rank_ic": metrics.get("rank_ic_mean"),
            "test_rank_ic_positive_rate": metrics.get("rank_ic_positive_rate"),
            "test_oos_r2": metrics.get("oos_r2_vs_historical_mean"),
            "test_mae": metrics.get("mae"),
            "test_qlike": metrics.get("qlike"),
            "prediction_std": metrics.get("prediction_std"),
            "unstable": bool(
                metrics.get("regressor") == "elasticnet_sgd"
                and (
                    abs(metrics.get("oos_r2_vs_historical_mean", 0.0)) > 1
                    or (metrics.get("qlike") is not None and metrics.get("qlike") > 100)
                    or metrics.get("prediction_std", 0.0) > 1
                )
            ),
            "result": str(result),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.main_root, "main") + _read(args.anchor_root, "anchor")
    frame = pd.DataFrame(rows)
    selected = []
    stable = frame[frame["regressor"].isin(["ridge", "huber_sgd"])].copy()
    for keys, group in stable.groupby(["variant", "axis", "factor", "target"], dropna=False):
        selected.append(group.loc[group["validation_ir"].idxmax()].to_dict())
    selected_frame = pd.DataFrame(selected)
    summaries = []
    for (variant, axis), group in selected_frame.groupby(["variant", "axis"]):
        summaries.append({
            "variant": variant,
            "axis": axis,
            "groups": int(len(group)),
            "test_ir_mean": float(group["test_ir"].mean()),
            "test_ir_median": float(group["test_ir"].median()),
            "test_ir_positive": int((group["test_ir"] > 0).sum()),
            "test_rank_ic_median": float(group["test_rank_ic"].median()),
            "test_oos_r2_median": float(group["test_oos_r2"].median()),
        })
    result = {
        "format_version": "aligned_prompt_regression_summary_v1",
        "counts": {"total": int(len(frame)), "main": int((frame.variant == "main").sum()), "anchor": int((frame.variant == "anchor").sum()), "unstable": int(frame.unstable.sum())},
        "all_results": rows,
        "stable_selection": selected,
        "axis_summary_stable_ridge_huber": summaries,
        "selection_note": "For each factor/target group, select Ridge or Huber by validation Rank-IC IR; test metrics are reported only after selection.",
        "unstable_note": "ElasticNet is flagged when test OOS R2 magnitude exceeds 1, QLIKE exceeds 100, or prediction std exceeds 1.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
