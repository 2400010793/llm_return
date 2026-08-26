"""Aggregate minimal-prompt token, semantic-cluster, and KNN results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.artifacts import atomic_json


REPRESENTATION_LABELS = {
    "group_pca_knn": "Group PCA + KNN",
    "cluster_augmented_knn": "Group PCA + cluster + KNN",
    "group_pca_logistic": "Group PCA + Logistic",
    "cluster_augmented_logistic": "Group PCA + cluster + Logistic",
    "cluster_majority": "Cluster majority",
    "majority": "Training majority",
}


def _rank_percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(
        ["model", "prompt_length", "variant", "target", "test_year"], sort=False,
    )[column].rank(method="average", pct=True, ascending=True)


def _accuracy_delta(frame: pd.DataFrame, left: str, right: str) -> float:
    labels = frame["label"].to_numpy(dtype=int)
    left_correct = (frame[left].to_numpy(dtype=float) >= 0.5) == labels
    right_correct = (frame[right].to_numpy(dtype=float) >= 0.5) == labels
    return float(left_correct.mean() - right_correct.mean())


def _plot_accuracy(overall: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = overall[overall["representation"].isin((
        "group_pca_knn", "cluster_augmented_knn",
        "group_pca_logistic", "cluster_augmented_logistic", "majority",
    ))].copy()
    if selected.empty:
        return
    selected["configuration"] = (
        selected["model"] + " / " + selected["variant"] + " / " + selected["target"]
    )
    configurations = selected["configuration"].drop_duplicates().tolist()
    representations = selected["representation"].drop_duplicates().tolist()
    x = np.arange(len(configurations))
    width = 0.8 / len(representations)
    figure, axis = plt.subplots(figsize=(max(12, len(configurations) * 1.35), 6))
    for index, representation in enumerate(representations):
        values = selected[selected["representation"] == representation].set_index(
            "configuration"
        )["weighted_accuracy"].reindex(configurations)
        axis.bar(
            x + (index - (len(representations) - 1) / 2) * width,
            values.to_numpy(), width=width, label=REPRESENTATION_LABELS[representation],
        )
    axis.set_xticks(x, configurations, rotation=35, ha="right")
    axis.set_ylabel("2018-2026 weighted Accuracy")
    axis.set_ylim(0.45, max(0.56, float(selected["weighted_accuracy"].max()) + 0.01))
    axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _cluster_bootstrap_delta(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    difference = (
        ((frame[left].to_numpy(dtype=float) >= 0.5) == frame["label"].to_numpy(dtype=int))
        .astype(np.float64)
        - ((frame[right].to_numpy(dtype=float) >= 0.5) == frame["label"].to_numpy(dtype=int))
        .astype(np.float64)
    )
    keys = (
        frame["stock_id"].astype(str) + "|"
        + pd.to_datetime(frame["entry_date"]).dt.strftime("%Y-%m-%d")
    )
    codes, uniques = pd.factorize(keys, sort=True)
    cluster_sum = np.bincount(codes, weights=difference, minlength=len(uniques))
    cluster_count = np.bincount(codes, minlength=len(uniques)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(uniques), size=len(uniques))
        draws[index] = cluster_sum[selected].sum() / cluster_count[selected].sum()
    return {
        "accuracy_delta": float(difference.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": int(samples),
        "stock_day_clusters": int(len(uniques)),
    }


def summarize(args: argparse.Namespace) -> dict[str, object]:
    metric_files = sorted(args.fold_root.glob("*/*/*/*/*/metrics.json"))
    requested_models = getattr(args, "model", None)
    if requested_models:
        selected_models = set(requested_models)
        metric_files = [
            path for path in metric_files
            if path.relative_to(args.fold_root).parts[0] in selected_models
        ]
    if args.strict and len(metric_files) != args.expected_folds:
        raise ValueError(
            f"expected {args.expected_folds} fold reports; found {len(metric_files)}"
        )
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in metric_files]
    token_files = [
        path.parent / "token_metrics.parquet" for path in metric_files
        if (path.parent / "token_metrics.parquet").is_file()
    ]
    group_files = [
        path.parent / "semantic_group_metrics.parquet" for path in metric_files
        if (path.parent / "semantic_group_metrics.parquet").is_file()
    ]
    tokens = (
        pd.concat((pd.read_parquet(path) for path in token_files), ignore_index=True)
        if token_files else pd.DataFrame()
    )
    groups = (
        pd.concat((pd.read_parquet(path) for path in group_files), ignore_index=True)
        if group_files else pd.DataFrame()
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not tokens.empty:
        tokens.to_parquet(args.output_root / "token_metrics_all.parquet", index=False)
    if not groups.empty:
        groups.to_parquet(args.output_root / "semantic_group_metrics_all.parquet", index=False)

    if tokens.empty:
        token_summary = pd.DataFrame()
    else:
        for metric in (
            "fisher", "context_cosine_distance", "context_standardized_l2", "mask_delta_l2",
        ):
            tokens[f"{metric}_percentile"] = _rank_percentile(tokens, metric)
        token_summary = tokens.groupby(
            ["model", "prompt_length", "variant", "target", "position_zero_based", "token", "semantic_group"],
            as_index=False,
        ).agg(
            fisher_mean=("fisher", "mean"),
            fisher_percentile_mean=("fisher_percentile", "mean"),
            context_cosine_mean=("context_cosine_distance", "mean"),
            context_cosine_percentile_mean=("context_cosine_distance_percentile", "mean"),
            context_standardized_l2_mean=("context_standardized_l2", "mean"),
            context_standardized_l2_percentile_mean=("context_standardized_l2_percentile", "mean"),
            mask_delta_l2_mean=("mask_delta_l2", "mean"),
            mask_delta_l2_percentile_mean=("mask_delta_l2_percentile", "mean"),
            years=("test_year", "nunique"),
        )
    token_summary.to_csv(args.output_root / "token_summary.csv", index=False)

    classification_rows = []
    for report in reports:
        for row in report.get("classification", []):
            classification_rows.append({
                "model": report["model"], "variant": report["variant"],
                "target": report["target"], "test_year": int(report["test_year"]),
                "selected_news_clusters": report.get("selected_news_clusters"),
                **row,
            })
    yearly = pd.DataFrame(classification_rows)
    yearly.to_csv(args.output_root / "classification_yearly.csv", index=False)
    overall_rows = []
    if not yearly.empty:
        for keys, part in yearly.groupby(["model", "variant", "target", "representation"]):
            model, variant, target, representation = keys
            weights = part["test_rows"].to_numpy(dtype=float)
            accuracy = part["test_accuracy"].to_numpy(dtype=float)
            overall_rows.append({
                "model": model, "variant": variant, "target": target,
                "representation": representation,
                "weighted_accuracy": float(np.average(accuracy, weights=weights)),
                "mean_year_accuracy": float(accuracy.mean()),
                "minimum_year_accuracy": float(accuracy.min()),
                "maximum_year_accuracy": float(accuracy.max()),
                "years": int(part["test_year"].nunique()),
                "test_rows": int(weights.sum()),
            })
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(args.output_root / "classification_overall.csv", index=False)

    comparisons = []
    for model, variant, target in overall[["model", "variant", "target"]].drop_duplicates().itertuples(index=False):
        selected = overall[
            (overall["model"] == model) & (overall["variant"] == variant)
            & (overall["target"] == target)
        ].set_index("representation")
        for augmented, plain in (
            ("cluster_augmented_knn", "group_pca_knn"),
            ("cluster_augmented_logistic", "group_pca_logistic"),
        ):
            if augmented in selected.index and plain in selected.index:
                comparisons.append({
                    "model": model, "variant": variant, "target": target,
                    "comparison": f"{augmented}_minus_{plain}",
                    "accuracy_delta": float(
                        selected.loc[augmented, "weighted_accuracy"]
                        - selected.loc[plain, "weighted_accuracy"]
                    ),
                })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(args.output_root / "cluster_incremental_effect.csv", index=False)

    paired_rows = []
    prediction_files = [path.parent / "predictions.parquet" for path in metric_files]
    prediction_files = [path for path in prediction_files if path.is_file()]
    predictions = []
    for path in prediction_files:
        report = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
        frame = pd.read_parquet(path)
        frame["model"] = report["model"]
        frame["variant"] = report["variant"]
        frame["target_name"] = report["target"]
        frame["test_year"] = int(report["test_year"])
        predictions.append(frame)
    if predictions:
        prediction_all = pd.concat(predictions, ignore_index=True)
        prediction_all.to_parquet(
            args.output_root / "predictions_all.parquet", index=False,
        )
        pairs = (
            ("cluster_augmented_knn", "group_pca_knn"),
            ("group_pca_knn", "majority"),
            ("cluster_augmented_knn", "majority"),
            ("cluster_augmented_logistic", "group_pca_logistic"),
            ("cluster_augmented_knn", "cluster_augmented_logistic"),
            ("group_pca_logistic", "majority"),
        )
        for keys, part in prediction_all.groupby(["model", "variant", "target_name"]):
            for left, right in pairs:
                result = _cluster_bootstrap_delta(part, left, right)
                yearly_delta = part.groupby("test_year").apply(
                    lambda values: _accuracy_delta(values, left, right),
                    include_groups=False,
                )
                paired_rows.append({
                    "model": keys[0], "variant": keys[1], "target": keys[2],
                    "left": left, "right": right,
                    "positive_years": int((yearly_delta > 0).sum()),
                    "negative_years": int((yearly_delta < 0).sum()),
                    **result,
                })
        platform_columns = (
            "group_pca_logistic", "cluster_augmented_logistic",
            "group_pca_knn", "cluster_augmented_knn",
        )
        manifest_rows = []
        platform_root = args.output_root / "simple_states"
        selected_predictions = prediction_all[
            prediction_all["target_name"].eq("next_day_return")
        ]
        for (model, variant), part in selected_predictions.groupby(
            ["model", "variant"], sort=True,
        ):
            missing = [column for column in platform_columns if column not in part]
            if missing:
                raise ValueError(f"platform predictions lack columns: {missing}")
            stock_day = part.groupby(
                ["stock_id", "entry_date"], as_index=False, sort=True, observed=True,
            )[list(platform_columns)].mean()
            prediction_path = platform_root / "predictions" / f"{model}_{variant}.parquet"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            stock_day.to_parquet(prediction_path, index=False)
            for column in platform_columns:
                factor_id = f"cninfo_minimal_short_{model}_{variant}_{column}"
                manifest_rows.append({
                    "task_id": len(manifest_rows),
                    "source_task_id": "minimal_prompt_short_v2",
                    "predictions": str(prediction_path.resolve()),
                    "factor_id": factor_id,
                    "output_dir": str((platform_root / "results" / factor_id).resolve()),
                    "prediction_column": column,
                    "duplicate_policy": "error",
                })
        simple_states_manifest = args.output_root / "simple_states_manifest.tsv"
        pd.DataFrame(manifest_rows).to_csv(
            simple_states_manifest, sep="\t", index=False,
        )
    else:
        simple_states_manifest = None
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(args.output_root / "paired_accuracy_bootstrap.csv", index=False)
    _plot_accuracy(overall, args.output_root / "classification_overall.png")

    if groups.empty:
        long_group_summary = pd.DataFrame()
    else:
        long_groups = groups[
            (groups["prompt_length"] == "long")
            & groups["semantic_group"].isin((
                "performance_cashflow", "orders_investment", "financing_equity",
                "regulation_litigation", "operating_risk", "industry_change",
                "company_governance",
            ))
        ]
        long_group_summary = long_groups.groupby(
            ["model", "variant", "target", "semantic_group"], as_index=False,
        ).agg(
            fisher_mean=("fisher_mean", "mean"),
            fisher_rank_best_mean=("fisher_rank_best", "mean"),
            context_cosine_mean=("context_cosine_mean", "mean"),
            context_standardized_l2_mean=("context_standardized_l2_mean", "mean"),
            mask_delta_l2_mean=("mask_delta_l2_mean", "mean"),
            years=("test_year", "nunique"),
        )
    long_group_summary.to_csv(args.output_root / "long_group_summary.csv", index=False)

    group_cluster_rows = []
    news_cluster_rows = []
    for path, report in zip(metric_files, reports):
        group_cluster_path = path.parent / "semantic_group_clusters.csv"
        if not group_cluster_path.is_file():
            group_cluster_path = path.parent / "long_group_clusters.csv"
        if group_cluster_path.is_file():
            cluster_frame = pd.read_csv(group_cluster_path)
            counts = cluster_frame["group_cluster"].value_counts()
            smallest = set(counts[counts == counts.min()].index.tolist())
            isolated = ",".join(sorted(
                cluster_frame[cluster_frame["group_cluster"].isin(smallest)]["semantic_group"]
            ))
            group_cluster_rows.append({
                "model": report["model"], "variant": report["variant"],
                "prompt_length": report["prompt_length"],
                "target": report["target"], "test_year": report["test_year"],
                "selected_k": int(cluster_frame["selected_k"].iloc[0]),
                "silhouette": float(cluster_frame["silhouette"].iloc[0]),
                "smallest_cluster_groups": isolated,
            })
        selection = pd.read_csv(path.parent / "news_cluster_k_selection.csv")
        chosen = selection[selection["k"] == int(report["selected_news_clusters"])].iloc[0]
        news_cluster_rows.append({
            "model": report["model"], "variant": report["variant"],
            "prompt_length": report["prompt_length"],
            "target": report["target"], "test_year": report["test_year"],
            "selected_k": int(report["selected_news_clusters"]),
            "silhouette": float(chosen["silhouette_mean"]),
            "ari_mean": float(chosen["ari_mean"]),
            "ari_min": float(chosen["ari_min"]),
        })
    group_cluster_folds = pd.DataFrame(group_cluster_rows, columns=[
        "model", "prompt_length", "variant", "target", "test_year",
        "selected_k", "silhouette", "smallest_cluster_groups",
    ])
    group_cluster_folds.to_csv(args.output_root / "long_group_cluster_folds.csv", index=False)
    if group_cluster_folds.empty:
        group_cluster_consensus = pd.DataFrame(columns=[
            "model", "prompt_length", "variant", "target",
            "smallest_cluster_groups", "years", "mean_k", "mean_silhouette",
        ])
    else:
        group_cluster_consensus = group_cluster_folds.groupby(
            ["model", "prompt_length", "variant", "target", "smallest_cluster_groups"],
            as_index=False,
        ).agg(
            years=("test_year", "nunique"), mean_k=("selected_k", "mean"),
            mean_silhouette=("silhouette", "mean"),
        )
    group_cluster_consensus.to_csv(
        args.output_root / "long_group_cluster_consensus.csv", index=False,
    )
    news_cluster_folds = pd.DataFrame(news_cluster_rows, columns=[
        "model", "prompt_length", "variant", "target", "test_year",
        "selected_k", "silhouette", "ari_mean", "ari_min",
    ])
    news_cluster_folds.to_csv(args.output_root / "news_cluster_stability_folds.csv", index=False)
    if news_cluster_folds.empty:
        news_cluster_summary = pd.DataFrame(columns=[
            "model", "prompt_length", "variant", "target", "mean_k", "min_k",
            "max_k", "mean_silhouette", "mean_ari", "minimum_ari",
        ])
    else:
        news_cluster_summary = news_cluster_folds.groupby(
            ["model", "prompt_length", "variant", "target"], as_index=False,
        ).agg(
            mean_k=("selected_k", "mean"), min_k=("selected_k", "min"),
            max_k=("selected_k", "max"), mean_silhouette=("silhouette", "mean"),
            mean_ari=("ari_mean", "mean"), minimum_ari=("ari_min", "min"),
        )
    news_cluster_summary.to_csv(args.output_root / "news_cluster_stability.csv", index=False)

    lines = [
        "# Minimal Prompt v2：Token 机制、语义组聚类与 KNN",
        "",
        f"- 完成 fold：{len(reports)}/{args.expected_folds}；token 行：{len(tokens):,}。",
        "- 所有指标只使用各测试年前的 6 年拟合窗口；PCA/聚类/KNN 参数由随后 2 年验证集选择，测试年不参与选择。",
        "- Accuracy 为收益方向预测；event_return_3d 对应 3 日标签，next_day_return 对应 1 日标签。",
        "",
        "## 聚类与 KNN 总体结果",
        "",
    ]
    if overall.empty:
        lines.append("Long fold 尚无分类结果。")
    else:
        display = overall.sort_values(
            ["target", "weighted_accuracy"], ascending=[True, False],
        ).copy()
        display["weighted_accuracy"] = display["weighted_accuracy"].map(lambda x: f"{x:.4%}")
        lines.extend(display.to_markdown(index=False).splitlines())
    lines.extend(["", "## 语义组层次聚类", ""])
    lines.extend(group_cluster_consensus.to_markdown(index=False).splitlines())
    lines.extend(["", "## 新闻聚类稳定性", ""])
    lines.extend(news_cluster_summary.to_markdown(index=False).splitlines())
    lines.extend(["", "## 聚类的增量效果", ""])
    if comparison_frame.empty:
        lines.append("尚无可比较结果。")
    else:
        shown = comparison_frame.copy()
        shown["accuracy_delta"] = shown["accuracy_delta"].map(lambda x: f"{x:+.4%}")
        lines.extend(shown.to_markdown(index=False).splitlines())
    lines.extend(["", "## 股票-交易日聚类 Bootstrap", ""])
    if paired_frame.empty:
        lines.append("尚无成对预测结果。")
    else:
        shown = paired_frame.copy()
        for column in ("accuracy_delta", "ci_low", "ci_high"):
            shown[column] = shown[column].map(lambda x: f"{x:+.4%}")
        lines.extend(shown.to_markdown(index=False).splitlines())
    lines.extend(["", "## 逐 token 排名", ""])
    token_groups = (
        token_summary.groupby(
            ["model", "prompt_length", "variant", "target"], sort=True,
        ) if not token_summary.empty else []
    )
    for (model, length, variant, target), part in token_groups:
        lines.append(f"### {model} / {variant} / {target}")
        shown = part.nlargest(5, "fisher_percentile_mean")[[
            "position_zero_based", "token", "semantic_group", "fisher_percentile_mean",
            "context_standardized_l2_percentile_mean", "mask_delta_l2_percentile_mean",
        ]].copy()
        for column in shown.columns[3:]:
            shown[column] = shown[column].map(lambda x: f"{x:.3f}")
        lines.extend(shown.to_markdown(index=False).splitlines())
        lines.append("")
    (args.output_root / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    audit = {
        "format_version": "minimal_prompt_knn_summary_v1",
        "folds_found": len(reports), "expected_folds": args.expected_folds,
        "token_rows": len(tokens), "classification_rows": len(yearly),
        "paired_bootstrap_rows": len(paired_frame),
        "complete": len(reports) == args.expected_folds,
        "outputs": {
            "report": str(args.output_root / "report_zh.md"),
            "overall": str(args.output_root / "classification_overall.csv"),
            "yearly": str(args.output_root / "classification_yearly.csv"),
            "simple_states_manifest": (
                str(simple_states_manifest) if simple_states_manifest else None
            ),
        },
    }
    atomic_json(args.output_root / "summary_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("roberta", "bge_m3"), action="append",
        help="Restrict aggregation to one or more models.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--expected-folds", type=int, default=144)
    args = parser.parse_args()
    print(json.dumps(summarize(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
