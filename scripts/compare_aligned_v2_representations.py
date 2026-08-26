"""Compare token-span and body-mean v2 regression results on matched tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read(root: Path, representation: str) -> list[dict]:
    rows = []
    for result in sorted(root.glob("*/result.json")):
        payload = json.loads(result.read_text(encoding="utf-8"))
        metrics = payload["results"][0]
        matrix = Path(payload["input"]["matrix"])
        parts = matrix.parts
        marker = next((index for index, value in enumerate(parts) if value.startswith("factors")), None)
        validation = metrics.get("validation_metrics", {})
        rows.append({
            "representation": representation,
            "axis": "anchor" if marker is None else parts[marker + 2],
            "factor": "anchor" if marker is None else parts[-1].removesuffix(".npy"),
            "target": metrics["target"],
            "regressor": metrics["regressor"],
            "validation_ir": validation.get("rank_ic_information_ratio"),
            "test_ir": metrics.get("rank_ic_information_ratio"),
            "test_rank_ic": metrics.get("rank_ic_mean"),
            "test_r2": metrics.get("oos_r2_vs_historical_mean"),
            "test_qlike": metrics.get("qlike"),
            "prediction_std": metrics.get("prediction_std"),
            "unstable": bool(
                abs(metrics.get("oos_r2_vs_historical_mean", 0.0)) > 1
                or (metrics.get("qlike") is not None and metrics.get("qlike") > 100)
                or metrics.get("prediction_std", 0.0) > 1
            ),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--body-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.token_root, "target_span_mean") + _read(args.body_root, "body_mean") + _read(args.anchor_root, "anchor_body_mean")
    frame = pd.DataFrame(rows)
    selected = []
    for representation, axis, factor, target in frame[frame.representation != "anchor_body_mean"].groupby(["representation", "axis", "factor", "target"], dropna=False).groups:
        group = frame[(frame.representation == representation) & (frame.axis == axis) & (frame.factor == factor) & (frame.target == target)]
        selected.append(group.loc[group.validation_ir.idxmax()].to_dict())
    selected_frame = pd.DataFrame(selected)
    summary = []
    for representation, group in selected_frame.groupby("representation"):
        for axis, axis_group in group.groupby("axis"):
            summary.append({
                "representation": representation,
                "axis": axis,
                "groups": int(len(axis_group)),
                "test_ir_median": float(axis_group.test_ir.median()),
                "test_ir_positive": int((axis_group.test_ir > 0).sum()),
                "test_rank_ic_median": float(axis_group.test_rank_ic.median()),
                "test_r2_median": float(axis_group.test_r2.median()),
            })
    matched = frame[frame.representation.isin(["target_span_mean", "body_mean"])].pivot_table(
        index=["axis", "factor", "target", "regressor"], columns="representation", values=["test_ir", "test_rank_ic", "test_r2"], aggfunc="first"
    )
    deltas = []
    for index, values in matched.iterrows():
        if ("test_ir", "target_span_mean") not in values or ("test_ir", "body_mean") not in values:
            continue
        deltas.append({
            "axis": index[0], "factor": index[1], "target": index[2], "regressor": index[3],
            "delta_test_ir": float(values[("test_ir", "target_span_mean")] - values[("test_ir", "body_mean")]),
            "delta_test_rank_ic": float(values[("test_rank_ic", "target_span_mean")] - values[("test_rank_ic", "body_mean")]),
            "delta_test_r2": float(values[("test_r2", "target_span_mean")] - values[("test_r2", "body_mean")]),
        })
    result = {
        "format_version": "aligned_v2_representation_comparison_v1",
        "counts": {"target_span_mean": int((frame.representation == "target_span_mean").sum()), "body_mean": int((frame.representation == "body_mean").sum()), "anchor_body_mean": int((frame.representation == "anchor_body_mean").sum()), "unstable": int(frame.unstable.sum())},
        "selected_summaries": summary,
        "matched_deltas": deltas,
        "selected_results": selected,
        "note": "Selection uses validation Rank-IC IR within each representation/factor/target group; test metrics are reported after selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
