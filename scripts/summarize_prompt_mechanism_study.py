"""Aggregate prompt mechanism folds into tables, plots, and a Chinese report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.classification import paired_classification_comparison


IDENTITY = ("model", "prompt_length", "variant", "target", "test_year")


def _load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    token_rows = []
    group_rows = []
    classification_rows = []
    prediction_rows = []
    reports = []
    for report_path in sorted(root.glob("*/*/*/*/test_*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        directory = report_path.parent
        identity = {
            "model": report["model"], "prompt_length": report["prompt_length"],
            "variant": report["variant"], "target": report["target"],
            "test_year": int(report["window"]["test_year"]),
        }
        reports.append(report)
        for name, collection in (
            ("token_metrics.parquet", token_rows),
            ("group_metrics.parquet", group_rows),
            ("classification_metrics.parquet", classification_rows),
            ("predictions.parquet", prediction_rows),
        ):
            frame = pd.read_parquet(directory / name)
            for key, value in identity.items():
                if key not in frame:
                    frame[key] = value
            collection.append(frame)
    if not reports:
        raise ValueError(f"no completed mechanism fold reports below {root}")
    return tuple(pd.concat(rows, ignore_index=True) for rows in (
        token_rows, group_rows, classification_rows, prediction_rows,
    )) + (reports,)


def _weighted_accuracy(group: pd.DataFrame) -> float:
    weights = pd.to_numeric(group["test_n"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(group["test_accuracy"], errors="coerce").to_numpy(dtype=float)
    return float(np.average(values, weights=weights))


def _classification_summary(classification: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "prompt_length", "variant", "target", "representation", "classifier"]
    rows = []
    for values, group in classification.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, values))
        row.update({
            "years": int(group["test_year"].nunique()),
            "weighted_accuracy": _weighted_accuracy(group),
            "mean_accuracy": float(group["test_accuracy"].mean()),
            "min_accuracy": float(group["test_accuracy"].min()),
            "max_accuracy": float(group["test_accuracy"].max()),
            "total_n": int(group["test_n"].sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["target", "weighted_accuracy"], ascending=[True, False],
    )


def _leaveout_evidence(predictions: pd.DataFrame, bootstrap: int) -> pd.DataFrame:
    logistic = predictions[predictions["classifier"].eq("logistic")].copy()
    keys = ["model", "prompt_length", "variant", "target"]
    rows = []
    for values, group in logistic.groupby(keys, sort=True):
        baseline = group[group["representation"].eq("full_prompt_mean")]
        if baseline.empty:
            continue
        baseline = baseline[["row_index", "probability"]].rename(
            columns={"probability": "baseline_probability"},
        )
        for representation in sorted(
            value for value in group["representation"].unique()
            if str(value).startswith("leaveout::")
        ):
            candidate = group[group["representation"].eq(representation)].merge(
                baseline, on="row_index", validate="one_to_one",
            )
            clusters = (
                pd.to_datetime(candidate["entry_date"]).dt.year.astype(str)
                + "|" + pd.to_datetime(candidate["entry_date"]).dt.strftime("%Y-%m-%d")
            )
            comparison = paired_classification_comparison(
                candidate["actual_return"], candidate["probability"],
                candidate["baseline_probability"], clusters,
                n_bootstrap=bootstrap, seed=42, metrics=("accuracy",),
            )
            yearly = candidate.assign(
                year=pd.to_datetime(candidate["entry_date"]).dt.year,
                candidate_correct=(candidate["probability"] > 0.5).eq(candidate["actual_return"] > 0),
                baseline_correct=(candidate["baseline_probability"] > 0.5).eq(candidate["actual_return"] > 0),
            ).groupby("year").agg(
                candidate_accuracy=("candidate_correct", "mean"),
                baseline_accuracy=("baseline_correct", "mean"),
            )
            contributions = yearly["baseline_accuracy"] - yearly["candidate_accuracy"]
            ci = comparison["clustered_95_ci"]["accuracy"]
            contribution = comparison["delta"]["accuracy"]
            row = dict(zip(keys, values))
            row.update({
                "semantic_group": representation.split("::", 1)[1],
                "years": int(len(yearly)),
                "years_positive_contribution": int((contributions > 0).sum()),
                "weighted_accuracy_contribution": float(contribution),
                "contribution_ci_low": float(ci[0]),
                "contribution_ci_high": float(ci[1]),
                "predictive_evidence": bool(
                    len(yearly) >= 9 and (contributions > 0).sum() >= 6
                    and contribution > 0 and ci[0] > 0
                ),
                "causal_evidence": False,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _cross_model_agreement(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["prompt_length", "variant", "target", "test_year"]
    for values, frame in groups.groupby(keys, sort=True):
        pivot = frame.pivot_table(
            index="semantic_group", columns="model", values="fisher_mean", aggfunc="mean",
        )
        for left_index, left in enumerate(pivot.columns):
            for right in pivot.columns[left_index + 1:]:
                aligned = pivot[[left, right]].dropna()
                correlation = spearmanr(aligned[left], aligned[right]).statistic if len(aligned) > 1 else np.nan
                rows.append({
                    **dict(zip(keys, values)), "model_left": left, "model_right": right,
                    "groups": len(aligned), "spearman": float(correlation),
                })
    return pd.DataFrame(rows)


def _heatmap(groups: pd.DataFrame, output: Path) -> None:
    values = groups[groups["prompt_length"].eq("long")].copy()
    if values.empty:
        values = groups.copy()
    if values.empty:
        return
    values["cell"] = values["model"] + "\n" + values["variant"] + "\n" + values["target"]
    pivot = values.pivot_table(
        index="semantic_group", columns="cell", values="fisher_mean", aggfunc="mean",
    )
    standardized = (pivot - pivot.mean(axis=0)) / pivot.std(axis=0).replace(0, np.nan)
    figure, axis = plt.subplots(figsize=(max(10, 0.8 * len(pivot.columns)), 7))
    image = axis.imshow(standardized.fillna(0), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=75, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index, fontsize=8)
    axis.set_title("Long-prompt semantic-group Fisher score (within-cell z-score)")
    figure.colorbar(image, ax=axis, label="z-score")
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


def _accuracy_plot(summary: pd.DataFrame, output: Path) -> None:
    logistic = summary[summary["classifier"].eq("logistic")].copy()
    top = logistic.sort_values("weighted_accuracy", ascending=False).groupby("target").head(12)
    top["label"] = (
        top["model"] + "/" + top["variant"] + "/" + top["representation"]
    )
    figure, axes = plt.subplots(1, len(top["target"].unique()), figsize=(16, 7), squeeze=False)
    for axis, (target, frame) in zip(axes[0], top.groupby("target", sort=True)):
        frame = frame.sort_values("weighted_accuracy")
        axis.barh(frame["label"], frame["weighted_accuracy"] * 100)
        axis.set_title(target); axis.set_xlabel("Weighted Accuracy (%)")
        axis.tick_params(axis="y", labelsize=7)
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


def _cluster_plots(input_root: Path, output_root: Path) -> int:
    count = 0
    for path in sorted(input_root.glob("*/long/*/*/test_2026/cluster_assignments.parquet")):
        frame = pd.read_parquet(path)
        report = json.loads((path.parent / "report.json").read_text(encoding="utf-8"))
        if len(frame) > 10_000:
            frame = frame.sample(10_000, random_state=42)
        figure, axis = plt.subplots(figsize=(7, 6))
        scatter = axis.scatter(
            frame["pca_1"], frame["pca_2"], c=frame["cluster"],
            s=5, alpha=0.45, cmap="tab20",
        )
        axis.set_title(
            f"{report['model']} / {report['variant']} / {report['target']} / 2026"
        )
        axis.set_xlabel("PCA 1"); axis.set_ylabel("PCA 2")
        figure.colorbar(scatter, ax=axis, label="cluster")
        destination = output_root / (
            f"cluster_{report['model']}_{report['variant']}_{report['target']}_2026.png"
        )
        figure.tight_layout(); figure.savefig(destination, dpi=170); plt.close(figure)
        count += 1
    return count


def _dendrograms(input_root: Path, output_root: Path) -> int:
    count = 0
    for path in sorted(input_root.glob("*/long/*/*/test_2026/token_linkage.npy")):
        report = json.loads((path.parent / "report.json").read_text(encoding="utf-8"))
        tokens = pd.read_parquet(path.parent / "token_metrics.parquet")["token"].astype(str).tolist()
        figure, axis = plt.subplots(figsize=(18, 7))
        dendrogram(np.load(path), labels=tokens, leaf_rotation=90, leaf_font_size=6, ax=axis)
        axis.set_title(
            f"Token response clustering: {report['model']} / {report['variant']} / {report['target']}"
        )
        figure.tight_layout()
        destination = output_root / (
            f"token_tree_{report['model']}_{report['variant']}_{report['target']}_2026.png"
        )
        figure.savefig(destination, dpi=170); plt.close(figure)
        count += 1
    return count


def _report_text(
    summary: pd.DataFrame, evidence: pd.DataFrame, agreement: pd.DataFrame,
    completed_folds: int,
) -> str:
    lines = [
        "# Prompt Token 表示机制与聚类研究", "",
        "## 审计范围", "",
        f"- 已完成 fold：{completed_folds}。",
        "- 两种标签严格分开：event_return_3d→event_return_3d 与 next_day_return→next_day_return。",
        "- 当前报告的因果结论状态：尚未建立；位置匹配 no-prompt 和关键词替换对照需在第一阶段筛选后运行。",
        "", "## Accuracy 最优结果", "",
        "| 标签 | 模型 | Prompt | 表示 | 分类器 | 加权 Accuracy | 年份 |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for target, frame in summary.groupby("target", sort=True):
        for _, row in frame.head(10).iterrows():
            lines.append(
                f"| {target} | {row['model']} | {row['variant']} | {row['representation']} | "
                f"{row['classifier']} | {row['weighted_accuracy']:.4%} | {int(row['years'])} |"
            )
    lines.extend(["", "## 语义组预测证据", ""])
    if evidence.empty:
        lines.append("尚无完整 leave-one-group-out 结果。")
    else:
        lines.extend([
            "判定标准为至少 6/9 年贡献为正、总体贡献为正且日期聚类 bootstrap 95% 下界大于 0。",
            "", "| 标签 | 模型/Prompt | 语义组 | 正贡献年份 | Accuracy贡献 | 95% CI | 预测证据 |",
            "|---|---|---|---:|---:|---:|---|",
        ])
        for _, row in evidence.sort_values("weighted_accuracy_contribution", ascending=False).head(30).iterrows():
            lines.append(
                f"| {row['target']} | {row['model']}/{row['variant']} | {row['semantic_group']} | "
                f"{int(row['years_positive_contribution'])}/{int(row['years'])} | "
                f"{row['weighted_accuracy_contribution']:.3%} | "
                f"[{row['contribution_ci_low']:.3%}, {row['contribution_ci_high']:.3%}] | "
                f"{'是' if row['predictive_evidence'] else '否'} |"
            )
    lines.extend(["", "## 跨模型稳定性", ""])
    if agreement.empty:
        lines.append("尚无完整跨模型语义组排名。")
    else:
        lines.append(
            f"语义组 Fisher 排名的模型两两 Spearman 中位数为 "
            f"{agreement['spearman'].median():.3f}；原始向量尺度未跨模型直接比较。"
        )
    lines.extend([
        "", "## 结论边界", "",
        "- 表示层：token 和语义组的上下文化变化、标签分离与聚类可直接由当前结果描述。",
        "- 预测层：只有滚动样本外 leave-one-group-out、top-8 或 cluster+body 的 Accuracy 差异可称为预测贡献。",
        "- 因果层：只有通过位置匹配 no-prompt、关键词等长替换及 placebo 后，才可称为 prompt 语义的优势来源。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tokens, groups, classification, predictions, reports = _load(args.input_root)
    summary = _classification_summary(classification)
    evidence = _leaveout_evidence(predictions, args.bootstrap)
    agreement = _cross_model_agreement(groups)
    tokens.to_parquet(args.output_root / "token_metrics_all.parquet", index=False)
    groups.to_parquet(args.output_root / "group_metrics_all.parquet", index=False)
    classification.to_parquet(args.output_root / "classification_by_year.parquet", index=False)
    summary.to_csv(args.output_root / "classification_summary.csv", index=False)
    evidence.to_csv(args.output_root / "semantic_group_evidence.csv", index=False)
    agreement.to_csv(args.output_root / "cross_model_agreement.csv", index=False)
    _heatmap(groups, args.output_root / "long_prompt_group_heatmap.png")
    _accuracy_plot(summary, args.output_root / "accuracy_top.png")
    cluster_plots = _cluster_plots(args.input_root, args.output_root)
    token_trees = _dendrograms(args.input_root, args.output_root)
    report = _report_text(summary, evidence, agreement, len(reports))
    (args.output_root / "report_zh.md").write_text(report, encoding="utf-8")
    audit = {
        "format_version": "prompt_mechanism_summary_v1", "completed_folds": len(reports),
        "expected_full_folds": 4 * 2 * 2 * 2 * 9,
        "cluster_plots": cluster_plots, "token_trees": token_trees,
        "causal_controls_complete": False,
        "outputs": sorted(path.name for path in args.output_root.iterdir()),
    }
    (args.output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
