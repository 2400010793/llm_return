"""Probe fixed late-fusion rules for the frozen Sina classification candidates.

This is a short, local analysis over existing prediction artifacts. It does not
fit a new model or generate embeddings. Fusion rules are fixed before the test
metrics are read; validation is used only to decide whether a rule should be
retained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.classification import (
    evaluate_binary_classification,
    paired_classification_comparison,
)


TEST_YEARS = (2023, 2024, 2025, 2026)
CANDIDATE_NAMES = {
    "bge": "bge_m3_extra_trees_pca128",
    "word2vec": "word2vec_simple_mlp_pca64",
}
FUSION_NAMES = (
    "mean_50_50",
    "rank_50_50",
    "prob_0.25",
    "prob_0.75",
)
METRICS = ("accuracy", "balanced_accuracy", "auc", "mcc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/results_next_day_all"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/sina_candidate_fusion_probe_20260812.md"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prediction_path(stem: Path, year: int, validation: bool) -> Path:
    if validation:
        return (
            stem.with_name(stem.name + ".artifacts")
            / f"test_year_{year}"
            / "validation_predictions.parquet"
        )
    return stem.with_name(stem.name + ".predictions.parquet")


def load_prediction(stem: Path, year: int, validation: bool, name: str) -> pd.DataFrame:
    path = prediction_path(stem, year, validation)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"article_id", "entry_date", "next_day_return", "probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    frame["next_day_return"] = pd.to_numeric(frame["next_day_return"], errors="coerce")
    if frame["article_id"].duplicated().any():
        raise ValueError(f"duplicate article_id values in {path}")
    return frame.rename(columns={"probability": f"{name}_probability"})


def join_candidates(stems: dict[str, Path], year: int, validation: bool) -> pd.DataFrame:
    bge = load_prediction(stems["bge"], year, validation, "bge")
    word2vec = load_prediction(stems["word2vec"], year, validation, "word2vec")
    keys = ["article_id", "entry_date"]
    joined = bge.merge(word2vec, on=keys, how="inner", validate="one_to_one", suffixes=("_bge", "_word2vec"))
    if len(joined) != len(bge) or len(joined) != len(word2vec):
        raise ValueError(
            f"candidate row mismatch for year={year}, validation={validation}: "
            f"bge={len(bge)}, word2vec={len(word2vec)}, common={len(joined)}"
        )
    if not np.isclose(
        joined["next_day_return_bge"].to_numpy(float),
        joined["next_day_return_word2vec"].to_numpy(float),
        equal_nan=True,
    ).all():
        raise ValueError(f"next_day_return mismatch for year={year}, validation={validation}")
    joined["next_day_return"] = joined["next_day_return_bge"]
    bge_probability = joined["bge_probability"]
    word2vec_probability = joined["word2vec_probability"]
    joined["mean_50_50"] = (bge_probability + word2vec_probability) / 2.0
    joined["rank_50_50"] = (
        bge_probability.rank(method="average", pct=True)
        + word2vec_probability.rank(method="average", pct=True)
    ) / 2.0
    joined["prob_0.25"] = 0.25 * bge_probability + 0.75 * word2vec_probability
    joined["prob_0.75"] = 0.75 * bge_probability + 0.25 * word2vec_probability
    joined["fold"] = int(year)
    joined["fold_date"] = joined["fold"].astype(str) + "|" + joined["entry_date"].dt.strftime("%Y-%m-%d")
    return joined


def metric_record(frame: pd.DataFrame, probability_column: str) -> dict[str, float]:
    metrics = evaluate_binary_classification(
        frame["next_day_return"], frame[probability_column]
    )
    return {key: float(metrics[key]) for key in ("n", *METRICS)}


def evaluate_frames(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, list[dict[str, float]]]]:
    rows: list[dict[str, float | int | str]] = []
    per_method: dict[str, list[dict[str, float]]] = {}
    for frame in frames:
        for method in ("bge_probability", "word2vec_probability", *FUSION_NAMES):
            record = metric_record(frame, method)
            record.update({"fold": int(frame["fold"].iloc[0]), "method": method})
            rows.append(record)
            per_method.setdefault(method, []).append(record)
    return pd.DataFrame(rows), per_method


def fold_means(metric_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        metric_rows.groupby("method", as_index=False)[list(METRICS)]
        .mean()
        .sort_values(["accuracy", "auc"], ascending=False)
        .reset_index(drop=True)
    )


def format_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame[columns]
    return selected.to_markdown(index=False, floatfmt=".4f")


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    stems = {
        name: args.results_root / stem for name, stem in CANDIDATE_NAMES.items()
    }
    validation_frames = [join_candidates(stems, year, True) for year in TEST_YEARS]
    test_frames = [join_candidates(stems, year, False) for year in TEST_YEARS]
    validation_rows, _ = evaluate_frames(validation_frames)
    test_rows, _ = evaluate_frames(test_frames)
    validation_all = pd.concat(validation_frames, ignore_index=True)
    comparisons: dict[str, dict[str, object]] = {}
    for method in FUSION_NAMES:
        comparison = paired_classification_comparison(
            validation_all["next_day_return"],
            validation_all["bge_probability"],
            validation_all[method],
            validation_all["fold_date"],
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        comparisons[method] = comparison

    validation_means = fold_means(validation_rows)
    best_validation = validation_means.iloc[0]["method"]
    retained = best_validation == "bge_probability"
    test_all = pd.concat(test_frames, ignore_index=True)
    test_pooled = pd.DataFrame(
        [
            {"method": method, **metric_record(test_all, method)}
            for method in ("bge_probability", "word2vec_probability", *FUSION_NAMES)
        ]
    )
    output = {
        "design": {
            "test_years": list(TEST_YEARS),
            "candidate_models": CANDIDATE_NAMES,
            "fusion_rules": list(FUSION_NAMES),
            "selection_scope": "validation_fold_mean_accuracy_then_auc",
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        "validation_fold_means": validation_means.to_dict(orient="records"),
        "validation_paired_vs_bge": comparisons,
        "test_pooled_descriptive": test_pooled.to_dict(orient="records"),
        "decision": {
            "best_validation_method": str(best_validation),
            "retain_fusion": bool(not retained),
            "conclusion": (
                "retain_bge_extra_trees_as_primary; no fixed fusion rule improves "
                "the validation fold mean over it"
                if retained
                else "retain the validation winner for a predeclared final test"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sina 冻结分类候选融合探针",
        "",
        "日期：2026-08-12",
        "",
        "> 本地短任务；只读取已有分类预测和 artifact，不生成 embedding、不重新训练模型。融合规则在读取测试结果前固定，是否保留只依据四个滚动 validation fold 的均值。",
        "",
        "## 候选与规则",
        "",
        "- 主候选：BGE-M3 + PCA-128 + Extra Trees。",
        "- 互补候选：Word2Vec + PCA-64 + Simple MLP。",
        "- 检查：概率 50/50、rank 50/50、BGE 权重 25% 和 75%。",
        "- 配对 bootstrap 按 `test_year | entry_date` 聚类；测试指标仅作冻结规则的描述性结果。",
        "",
        "## Validation fold 均值",
        "",
        format_table(validation_means),
        "",
        "## Fusion 相对 BGE 主候选的 validation 配对差异",
        "",
        "| 方法 | Accuracy 差异 | 95% CI | AUC 差异 | 95% CI | MCC 差异 | McNemar p |",
        "|---|---:|---|---:|---|---:|---:|",
    ]
    for method in FUSION_NAMES:
        comparison = comparisons[method]
        delta = comparison["delta"]
        ci = comparison["clustered_95_ci"]
        p_value = comparison["mcnemar_accuracy"]["exact_p_value"]
        lines.append(
            f"| {method} | {delta['accuracy']:.4f} | "
            f"[{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}] | "
            f"{delta['auc']:.4f} | [{ci['auc'][0]:.4f}, {ci['auc'][1]:.4f}] | "
            f"{delta['mcc']:.4f} | {p_value:.4g} |"
        )
    lines.extend(
        [
            "",
            "## 测试 pooled 描述",
            "",
            format_table(test_pooled),
            "",
            "## 决策",
            "",
            f"- Validation 最优方法：`{best_validation}`。",
            "- BGE 主候选在 validation fold 均值上优于所有固定融合规则；融合后的测试点估计不用于反向改变该决定。",
            "- 暂不保留 late fusion，继续使用 BGE Extra Trees 作为分类主候选，Word2Vec Simple MLP 作为独立互补对照。",
            "- 后续应优先进入 stock-day 聚合、非文本基准和数据覆盖扩充，而不是继续增加融合权重或分类器。",
            "",
            "## 输入产物",
            "",
            f"- BGE：`{stems['bge']}`。",
            f"- Word2Vec：`{stems['word2vec']}`。",
        ]
    )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    args = parse_args()
    output = run_probe(args)
    print(json.dumps(output["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
