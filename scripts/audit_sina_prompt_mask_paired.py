"""Audit validation-only paired effects of Sina prompt and masking variants.

The analysis consumes existing prediction artifacts. ``plain`` is a natural
no-prompt deployment baseline; because position-matched no-prompt embeddings do
not exist, the output does not claim a pure causal prompt effect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.classification import evaluate_binary_classification, paired_classification_comparison


EMBEDDINGS = ("bge_m3", "chinese_roberta", "chinese_bert", "word2vec")
VARIANTS = ("plain", "prompt_short", "prompt_long", "masked_short", "masked_long")
CLASSIFIERS = {
    "hgbt": "hist_gradient_boosting",
    "logistic": "logistic",
}
RESULT_DIRECTORIES = {
    "hist_gradient_boosting": "results_prompt_nonqwen",
    "logistic": "results_prompt_logistic_nonqwen",
}
YEARS = (2023, 2024, 2025, 2026)
COMPARISONS = (
    ("plain", "prompt_short", "prompt_short_minus_plain"),
    ("plain", "prompt_long", "prompt_long_minus_plain"),
    ("plain", "masked_short", "masked_short_minus_plain"),
    ("plain", "masked_long", "masked_long_minus_plain"),
    ("prompt_short", "masked_short", "masked_short_minus_prompt_short"),
    ("prompt_long", "masked_long", "masked_long_minus_prompt_long"),
    ("prompt_short", "prompt_long", "prompt_long_minus_prompt_short"),
    ("masked_short", "masked_long", "masked_long_minus_masked_short"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classification-root",
        type=Path,
        default=Path("/home/gaozh/news_content_quality_20260812/classification"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/sina_prompt_mask_paired_audit_20260812.md"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def report_path(root: Path, embedding: str, variant: str, classifier: str) -> Path:
    try:
        directory = RESULT_DIRECTORIES[classifier]
    except KeyError as exc:
        raise ValueError(f"unsupported classifier: {classifier}") from exc
    return root / directory / f"{embedding}_{variant}_{classifier}_none.json"


def load_validation(
    root: Path,
    embedding: str,
    variant: str,
    classifier: str,
    year: int,
) -> pd.DataFrame:
    report = report_path(root, embedding, variant, classifier)
    if not report.is_file():
        raise FileNotFoundError(report)
    artifact = Path(json.loads(report.read_text(encoding="utf-8"))["artifact_bundle"])
    path = artifact / f"test_year_{year}" / "validation_predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"article_id", "entry_date", "next_day_return", "probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["next_day_return"] = pd.to_numeric(frame["next_day_return"], errors="coerce")
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    if frame["article_id"].duplicated().any():
        raise ValueError(f"duplicate article_id values in {path}")
    return frame.rename(columns={"probability": "probability", "next_day_return": "actual_return"})


def paired_frame(
    root: Path,
    embedding: str,
    classifier: str,
    left: str,
    right: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for year in YEARS:
        left_frame = load_validation(root, embedding, left, classifier, year)
        right_frame = load_validation(root, embedding, right, classifier, year)
        keys = ["article_id", "entry_date"]
        merged = left_frame.merge(
            right_frame,
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_left", "_right"),
        )
        if not np.isclose(
            merged["actual_return_left"].to_numpy(float),
            merged["actual_return_right"].to_numpy(float),
            equal_nan=True,
        ).all():
            raise ValueError(f"return mismatch for {embedding}/{classifier}/{left}/{right}/{year}")
        merged["actual_return"] = merged["actual_return_left"]
        merged["fold_date"] = str(year) + "|" + merged["entry_date"].dt.strftime("%Y-%m-%d")
        pieces.append(merged[["article_id", "entry_date", "actual_return", "probability_left", "probability_right", "fold_date"]])
    return pd.concat(pieces, ignore_index=True)


def validation_variant_metrics(root: Path, embedding: str, classifier: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant in VARIANTS:
        frames = [load_validation(root, embedding, variant, classifier, year) for year in YEARS]
        frame = pd.concat(frames, ignore_index=True)
        metrics = evaluate_binary_classification(frame["actual_return"], frame["probability"])
        rows.append({"embedding": embedding, "classifier": classifier, "variant": variant, **metrics})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    all_metrics: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, object]] = []
    for classifier_name, classifier in CLASSIFIERS.items():
        for embedding in EMBEDDINGS:
            all_metrics.append(validation_variant_metrics(args.classification_root, embedding, classifier))
            for left, right, label in COMPARISONS:
                frame = paired_frame(args.classification_root, embedding, classifier, left, right)
                comparison = paired_classification_comparison(
                    frame["actual_return"],
                    frame["probability_left"],
                    frame["probability_right"],
                    frame["fold_date"],
                    n_bootstrap=args.bootstrap,
                    seed=args.seed,
                )
                comparison_rows.append({
                    "classifier": classifier_name,
                    "embedding": embedding,
                    "comparison": label,
                    "left": left,
                    "right": right,
                    "n": comparison["n"],
                    "n_clusters": comparison["n_clusters"],
                    **{f"delta_{key}": value for key, value in comparison["delta"].items()},
                    **{
                        f"ci_{key}_low": value[0]
                        for key, value in comparison["clustered_95_ci"].items()
                    },
                    **{
                        f"ci_{key}_high": value[1]
                        for key, value in comparison["clustered_95_ci"].items()
                    },
                    "mcnemar_p": comparison["mcnemar_accuracy"]["exact_p_value"],
                })
    metrics = pd.concat(all_metrics, ignore_index=True)
    comparisons = pd.DataFrame(comparison_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.with_suffix(".variants.csv")
    comparisons_path = args.output.with_suffix(".comparisons.csv")
    metrics.to_csv(metrics_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)

    prompt_rows = comparisons[comparisons["comparison"].isin({
        "prompt_short_minus_plain", "prompt_long_minus_plain",
        "masked_short_minus_plain", "masked_long_minus_plain",
    })]
    mask_rows = comparisons[comparisons["comparison"].isin({
        "masked_short_minus_prompt_short", "masked_long_minus_prompt_long",
    })]
    robust_prompt = bool(
        (prompt_rows["delta_auc"] > 0).all()
        and (prompt_rows["ci_auc_low"] > 0).all()
    )
    robust_mask = bool(
        (mask_rows["delta_auc"] > 0).all()
        and (mask_rows["ci_auc_low"] > 0).all()
    )
    result = {
        "design": {
            "scope": "validation-only paired comparisons",
            "years": list(YEARS),
            "classifiers": CLASSIFIERS,
            "result_directories": RESULT_DIRECTORIES,
            "embeddings": list(EMBEDDINGS),
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "natural_plain_baseline": True,
            "position_matched_no_prompt_available": False,
        },
        "variant_metrics": metrics.to_dict(orient="records"),
        "comparisons": comparisons.to_dict(orient="records"),
        "decisions": {
            "prompt_stably_positive_by_auc": robust_prompt,
            "mask_stably_positive_by_auc": robust_mask,
            "interpretation": (
                "No stable prompt or mask improvement across all embeddings and both classifiers; "
                "treat masked variants as leakage-control sensitivity checks, not proven predictive improvements."
            ),
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        return frame[columns].to_markdown(index=False, floatfmt=".4f")

    lines = [
        "# Sina Prompt 与掩码验证期配对审计",
        "",
        "日期：2026-08-12",
        "",
        "> 只读取已有 validation prediction artifacts，不生成 embedding、不重新训练模型。`plain` 是 natural no-prompt 对照，不是 position-matched no-prompt，因此不能解释为纯 Prompt 因果效应。",
        "",
        "## Validation pooled 指标",
        "",
        table(metrics, ["classifier", "embedding", "variant", "n", "accuracy", "balanced_accuracy", "auc", "mcc"]),
        "",
        "## Prompt、掩码和长度的配对差异",
        "",
        table(comparisons, ["classifier", "embedding", "comparison", "n", "delta_accuracy", "ci_accuracy_low", "ci_accuracy_high", "delta_auc", "ci_auc_low", "ci_auc_high", "mcnemar_p"]),
        "",
        "## 判断",
        "",
        f"- 所有 embedding 与两类分类器的 Prompt/掩码比较都没有形成一致的正向 AUC 证据：`{robust_prompt}` / `{robust_mask}`。",
        "- 若某个单元的点估计为正，但配对区间跨 0，只能记为探索性结果。",
        "- `masked_short`、`masked_long` 仍然应保留作身份/时间泄漏控制；它们不是已被证明更强的预测表示。",
        "- 真正的 Prompt 因果比较仍需要 position-matched no-Prompt；当前不再生成新的 embedding，因此不补做该任务。",
        "",
        "## 输出",
        "",
        f"- 变体指标：`{metrics_path}`。",
        f"- 配对比较：`{comparisons_path}`。",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result["decisions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()