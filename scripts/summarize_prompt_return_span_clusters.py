"""Summarize exact `收益` token news-clustering folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.artifacts import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-folds", type=int, default=144)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    files = sorted(args.fold_root.glob("*/*/*/*/*/metrics.json"))
    if args.strict and len(files) != args.expected_folds:
        raise ValueError(
            f"expected {args.expected_folds} return-span folds; found {len(files)}"
        )
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    frame = pd.DataFrame([{k: value for k, value in row.items() if k != "window"} for row in rows])
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_root / "fold_results.csv", index=False)
    summary_rows = []
    for keys, part in frame.groupby(["model", "prompt_length", "variant", "target"]):
        weights = part["test_rows"].to_numpy(dtype=float)
        k_counts = part["selected_k"].value_counts()
        diagnostics = pd.json_normalize(part["selected_k_diagnostics"])
        summary_rows.append({
            "model": keys[0], "prompt_length": keys[1], "variant": keys[2],
            "target": keys[3], "modal_k": int(k_counts.index[0]),
            "modal_k_years": int(k_counts.iloc[0]),
            "mean_k": float(part["selected_k"].mean()),
            "min_k": int(part["selected_k"].min()),
            "max_k": int(part["selected_k"].max()),
            "mean_silhouette": float(diagnostics["silhouette_mean"].mean()),
            "mean_ari": float(diagnostics["ari_mean"].mean()),
            "minimum_ari": float(diagnostics["ari_min"].min()),
            "cluster_accuracy": float(np.average(part["cluster_accuracy"], weights=weights)),
            "majority_accuracy": float(np.average(part["majority_accuracy"], weights=weights)),
            "accuracy_delta": float(np.average(part["accuracy_delta"], weights=weights)),
            "mean_positive_rate_spread": float(part["positive_rate_spread"].mean()),
            "mean_return_spread": float(part["mean_return_spread"].mean()),
            "test_rows": int(weights.sum()), "years": int(part["test_year"].nunique()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_root / "overall.csv", index=False)

    extreme_rows = []
    correlation_rows = []
    all_cluster_rows = []
    for path, report in zip(files, rows):
        target = report["target"]
        prediction = pd.read_parquet(path.parent / "predictions.parquet")
        clusters = prediction.groupby("cluster").agg(
            rows=("label", "size"),
            train_positive_probability=("cluster_probability", "first"),
            test_positive_rate=("label", "mean"),
            test_mean_return=(target, "mean"),
            test_median_return=(target, "median"),
        ).reset_index().sort_values("train_positive_probability")
        for cluster_rank, (_, cluster_row) in enumerate(clusters.iterrows(), start=1):
            all_cluster_rows.append({
                "model": report["model"], "prompt_length": report["prompt_length"],
                "variant": report["variant"], "target": target,
                "test_year": report["test_year"], "selected_k": report["selected_k"],
                "train_probability_rank": cluster_rank, **cluster_row.to_dict(),
            })
        for rank, selected in (
            ("lowest_train_probability", clusters.iloc[0]),
            ("highest_train_probability", clusters.iloc[-1]),
        ):
            extreme_rows.append({
                "model": report["model"], "prompt_length": report["prompt_length"],
                "variant": report["variant"], "target": target,
                "test_year": report["test_year"], "rank": rank, **selected.to_dict(),
            })
        correlation_rows.append({
            "model": report["model"], "prompt_length": report["prompt_length"],
            "variant": report["variant"], "target": target,
            "test_year": report["test_year"],
            "rho_positive_rate": float(spearmanr(
                clusters["train_positive_probability"], clusters["test_positive_rate"],
            ).statistic),
            "rho_mean_return": float(spearmanr(
                clusters["train_positive_probability"], clusters["test_mean_return"],
            ).statistic),
        })
    extremes = pd.DataFrame(extreme_rows)
    extremes.to_csv(args.output_root / "ranked_cluster_extremes_by_year.csv", index=False)
    pd.DataFrame(all_cluster_rows).to_csv(
        args.output_root / "all_cluster_returns_by_year.csv", index=False,
    )
    ranked_rows = []
    for keys, part in extremes.groupby(
        ["model", "prompt_length", "variant", "target", "rank"],
    ):
        weights = part["rows"].to_numpy(dtype=float)
        ranked_rows.append({
            "model": keys[0], "prompt_length": keys[1], "variant": keys[2],
            "target": keys[3], "rank": keys[4], "rows": int(weights.sum()),
            "train_positive_probability": float(np.average(
                part["train_positive_probability"], weights=weights,
            )),
            "test_positive_rate": float(np.average(
                part["test_positive_rate"], weights=weights,
            )),
            "test_mean_return": float(np.average(
                part["test_mean_return"], weights=weights,
            )),
            "median_of_yearly_cluster_medians": float(part["test_median_return"].median()),
        })
    ranked = pd.DataFrame(ranked_rows)
    ranked.to_csv(args.output_root / "ranked_cluster_extremes_overall.csv", index=False)
    correlations = pd.DataFrame(correlation_rows)
    correlation_summary = correlations.groupby(
        ["model", "prompt_length", "variant", "target"], as_index=False,
    ).agg(
        mean_rho_positive_rate=("rho_positive_rate", "mean"),
        mean_rho_return=("rho_mean_return", "mean"),
        positive_rho_years=("rho_mean_return", lambda values: int((values > 0).sum())),
    )
    correlation_summary.to_csv(args.output_root / "cluster_rank_correlation.csv", index=False)
    factor_root = args.output_root / "simple_states" / "predictions"
    result_root = args.output_root / "simple_states" / "results"
    factor_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for keys, part in pd.DataFrame(all_cluster_rows).groupby(
        ["model", "prompt_length", "variant", "target"], sort=True,
    ):
        if keys[3] != "next_day_return":
            continue
        prediction_parts = []
        selected_files = [
            (path, report) for path, report in zip(files, rows)
            if (
                report["model"], report["prompt_length"],
                report["variant"], report["target"],
            ) == keys
        ]
        for path, _ in selected_files:
            prediction = pd.read_parquet(path.parent / "predictions.parquet")
            prediction_parts.append(
                prediction.groupby(["stock_id", "entry_date"], as_index=False).agg(
                    cluster_probability=("cluster_probability", "mean"),
                )
            )
        factor = pd.concat(prediction_parts, ignore_index=True)
        factor_id = "_".join(("return_span_cluster", *map(str, keys[:3])))
        prediction_path = factor_root / f"{factor_id}.parquet"
        factor.to_parquet(prediction_path, index=False)
        manifest_rows.append({
            "task_id": len(manifest_rows),
            "source_task_id": "return_span_cluster",
            "predictions": str(prediction_path.resolve()),
            "factor_id": factor_id,
            "output_dir": str((result_root / factor_id).resolve()),
            "prediction_column": "cluster_probability",
            "duplicate_policy": "error",
        })
    pd.DataFrame(manifest_rows).to_csv(
        args.output_root / "simple_states_manifest.tsv", sep="\t", index=False,
    )
    lines = [
        "# 精确“收益”Token Embedding 新闻聚类",
        "",
        f"- 完成：{len(frame)}/{args.expected_folds} folds。",
        "- 每篇新闻只保留 prompt 中精确字符区间“收益”的上下文化 embedding；不包含“未来”和七个 Long 语义组。",
        "- 历史 6 年拟合 PCA/聚类，随后 2 年用 silhouette 与五 seed ARI 选择 k=2..12，测试年只分配簇。",
        "",
    ]
    shown = summary.copy()
    for column in ("cluster_accuracy", "majority_accuracy", "accuracy_delta"):
        shown[column] = shown[column].map(lambda value: f"{value:+.4%}")
    lines.extend(shown.to_markdown(index=False).splitlines())
    lines.extend(["", "## 按训练期概率排序的样本外极端簇", ""])
    ranked_shown = ranked.copy()
    for column in (
        "train_positive_probability", "test_positive_rate", "test_mean_return",
        "median_of_yearly_cluster_medians",
    ):
        ranked_shown[column] = ranked_shown[column].map(lambda value: f"{value:+.4%}")
    lines.extend(ranked_shown.to_markdown(index=False).splitlines())
    lines.extend(["", "## 训练期簇排序与测试期收益的相关性", ""])
    lines.extend(correlation_summary.to_markdown(index=False).splitlines())
    (args.output_root / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    audit = {
        "format_version": "prompt_return_span_cluster_summary_v1",
        "folds": len(frame), "expected_folds": args.expected_folds,
        "complete": len(frame) == args.expected_folds,
        "simple_states_tasks": len(manifest_rows),
    }
    atomic_json(args.output_root / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
