"""Summarize encoder controls and decide whether prompt semantics have causal support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.classification import paired_classification_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    metrics = []
    predictions = []
    reports = []
    for report_path in sorted(args.input_root.glob("*/*/*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports.append(report)
        metric = pd.read_parquet(report_path.parent / "metrics.parquet")
        prediction = pd.read_parquet(report_path.parent / "predictions.parquet")
        for key in ("control_id", "semantic_group", "is_placebo"):
            metric[key] = report["spec"].get(key)
            prediction[key] = report["spec"].get(key)
        for key in ("model", "source_variant"):
            metric[key] = report["spec"][key]
            prediction[key] = report["spec"][key]
        predictions.append(prediction); metrics.append(metric)
    if not reports:
        raise ValueError("no completed control classification reports")
    metrics_frame = pd.concat(metrics, ignore_index=True)
    predictions_frame = pd.concat(predictions, ignore_index=True)
    keys = ["model", "source_variant", "target", "feature", "control_id", "semantic_group", "is_placebo"]
    evidence = []
    for values, frame in predictions_frame.groupby(keys, dropna=False, sort=True):
        clusters = pd.to_datetime(frame["entry_date"]).dt.strftime("%Y-%m-%d")
        comparison = paired_classification_comparison(
            frame["actual_return"], frame["original_probability"],
            frame["control_probability"], clusters,
            n_bootstrap=args.bootstrap, seed=42, metrics=("accuracy",),
        )
        yearly = metrics_frame
        for key, value in zip(keys, values):
            yearly = yearly[yearly[key].fillna("__NA__").eq("__NA__" if pd.isna(value) else value)]
        delta = comparison["delta"]["accuracy"]
        ci = comparison["clustered_95_ci"]["accuracy"]
        evidence.append({
            **dict(zip(keys, values)), "years": int(yearly["test_year"].nunique()),
            "years_control_hurts": int((yearly["control_minus_original"] < 0).sum()),
            "control_minus_original": delta, "delta_ci_low": ci[0], "delta_ci_high": ci[1],
            "hurts_with_ci": bool(delta < 0 and ci[1] < 0),
        })
    evidence_frame = pd.DataFrame(evidence)

    placebo_rows = []
    for _, ablation in evidence_frame[
        evidence_frame["control_id"].astype(str).str.startswith("ablate_")
    ].iterrows():
        placebo_id = str(ablation["control_id"]).replace("ablate_", "placebo_", 1)
        match = evidence_frame[
            evidence_frame["model"].eq(ablation["model"])
            & evidence_frame["source_variant"].eq(ablation["source_variant"])
            & evidence_frame["target"].eq(ablation["target"])
            & evidence_frame["feature"].eq(ablation["feature"])
            & evidence_frame["control_id"].eq(placebo_id)
        ]
        if len(match) != 1:
            continue
        pair = predictions_frame[
            predictions_frame["model"].eq(ablation["model"])
            & predictions_frame["source_variant"].eq(ablation["source_variant"])
            & predictions_frame["target"].eq(ablation["target"])
            & predictions_frame["feature"].eq(ablation["feature"])
            & predictions_frame["control_id"].isin([ablation["control_id"], placebo_id])
        ]
        left = pair[pair["control_id"].eq(placebo_id)][
            ["row_index", "entry_date", "actual_return", "control_probability"]
        ].rename(columns={"control_probability": "placebo_probability"})
        right = pair[pair["control_id"].eq(ablation["control_id"])][
            ["row_index", "control_probability"]
        ].rename(columns={"control_probability": "ablation_probability"})
        aligned = left.merge(right, on="row_index", validate="one_to_one")
        comparison = paired_classification_comparison(
            aligned["actual_return"], aligned["placebo_probability"],
            aligned["ablation_probability"], aligned["entry_date"],
            n_bootstrap=args.bootstrap, seed=42, metrics=("accuracy",),
        )
        delta = comparison["delta"]["accuracy"]
        ci = comparison["clustered_95_ci"]["accuracy"]
        placebo_rows.append({
            "model": ablation["model"], "source_variant": ablation["source_variant"],
            "target": ablation["target"], "feature": ablation["feature"],
            "semantic_group": ablation["semantic_group"],
            "ablation_minus_placebo": delta,
            "delta_ci_low": ci[0], "delta_ci_high": ci[1],
            "ablation_worse_with_ci": bool(delta < 0 and ci[1] < 0),
            "causal_support": bool(
                ablation["years"] >= 9 and ablation["years_control_hurts"] >= 6
                and ablation["hurts_with_ci"] and delta < 0 and ci[1] < 0
            ),
        })
    placebo_frame = pd.DataFrame(placebo_rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_parquet(args.output_root / "control_metrics_by_year.parquet", index=False)
    evidence_frame.to_csv(args.output_root / "control_vs_original.csv", index=False)
    placebo_frame.to_csv(args.output_root / "ablation_vs_placebo.csv", index=False)
    lines = [
        "# Prompt 位置匹配与语义替换对照", "",
        f"已完成 {len(reports)} 个 control×target 分类任务。", "",
        "| 标签 | 模型/Prompt | 特征 | 语义组 | Ablation-Placebo Accuracy | 95% CI | 因果支持 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for _, row in placebo_frame.iterrows():
        lines.append(
            f"| {row['target']} | {row['model']}/{row['source_variant']} | {row['feature']} | "
            f"{row['semantic_group']} | {row['ablation_minus_placebo']:.3%} | "
            f"[{row['delta_ci_low']:.3%}, {row['delta_ci_high']:.3%}] | "
            f"{'是' if row['causal_support'] else '否'} |"
        )
    (args.output_root / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = {
        "format_version": "prompt_mechanism_control_summary_v1",
        "classification_reports": len(reports),
        "causal_supported_rows": int(placebo_frame.get("causal_support", pd.Series(dtype=bool)).sum()),
    }
    (args.output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
