#!/usr/bin/env python3
"""Freeze prompt/span/Qwen representation RankICs used by the PDF report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSITIVE = Path(
    "/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/"
    "single_stock_cninfo_v1/prompt_positive_v1/summary"
)
DEFAULT_QWEN = Path(
    "/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/"
    "single_stock_cninfo_v1/qwen3_prompt_return_regression_v1/"
    "pca_ablation_v1/summary/overall_metrics.csv"
)
DEFAULT_OUTPUT = ROOT / "reports/comprehensive_research_report/audits/prompt_representations"
PROMPTS = {
    "profit": ("盈利", "盈利"),
    "return": ("收益", "收益"),
    "excess_return": ("超额收益", "超额"),
    "loss": ("亏损", "亏损"),
}
MODELS = {"roberta": "RoBERTa", "bge_m3": "BGE-M3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-summary", type=Path, default=DEFAULT_POSITIVE)
    parser.add_argument("--qwen-overall", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def rankic_by_method(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["method"]: float(row["rankic"]) for row in csv.DictReader(handle)}


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    four_prompt = []
    model_summary = []
    for model_key, model_name in MODELS.items():
        prompt_values = rankic_by_method(
            args.positive_summary
            / f"prompt_late_fusion_{model_key}_masked_prompt_mean_yearly_overall.csv"
        )
        span_values = rankic_by_method(
            args.positive_summary
            / f"prompt_late_fusion_{model_key}_masked_short_word_span_yearly_overall.csv"
        )
        for method, (prompt, historical_span) in PROMPTS.items():
            four_prompt.append(
                {
                    "model": model_name,
                    "prompt": prompt,
                    "historical_direction_span": historical_span,
                    "prompt_mean_rankic": prompt_values[method],
                    "direction_span_rankic": span_values[method],
                    "prompt_minus_span": prompt_values[method] - span_values[method],
                }
            )
        selected = [row for row in four_prompt if row["model"] == model_name]
        prompt_average = mean(float(row["prompt_mean_rankic"]) for row in selected)
        span_average = mean(float(row["direction_span_rankic"]) for row in selected)
        model_summary.append(
            {
                "model": model_name,
                "prompt_mean_rankic": prompt_average,
                "direction_span_rankic": span_average,
                "prompt_minus_span": prompt_average - span_average,
            }
        )

    with args.qwen_overall.open(encoding="utf-8", newline="") as handle:
        qwen_all = list(csv.DictReader(handle))
    wanted = {"article_mean", "prompt_mean", "return_token"}
    qwen = [
        {
            "representation": row["representation"],
            "rankic": float(row["rank_ic"]),
            "positive_years": int(row["positive_years"]),
            "test_stock_days": int(row["test_stock_days"]),
        }
        for row in qwen_all
        if row["variant"] == "masked_short"
        and row["reducer"] == "pca32"
        and row["representation"] in wanted
    ]
    if {row["representation"] for row in qwen} != wanted:
        raise ValueError("Qwen masked PCA32 representation set is incomplete")
    article_rankic = next(row["rankic"] for row in qwen if row["representation"] == "article_mean")
    for row in qwen:
        row["delta_vs_article_mean"] = row["rankic"] - article_rankic

    for name, rows in (
        ("four_prompt_masked_rankic.csv", four_prompt),
        ("qwen_masked_pca32_rankic.csv", qwen),
    ):
        with (args.output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "four_prompt_protocol": "Sina legacy panel; masked_short; PCA128+Ridge; 6+2+1; next_day_return",
        "historical_span_warning": "超额收益 uses the historical 超额 span, not the full 超额收益 target",
        "four_prompt_model_averages": model_summary,
        "qwen_protocol": "Sina; 分析股票收益。; masked_short; PCA32+Ridge; 6+2+1",
        "qwen": qwen,
        "true_body_mean_status": (
            "Saved in RoBERTa/BGE-M3 embedding outputs, but no completed same-sample "
            "historical four-prompt rolling regression; historical 0.04391/0.03307 values are full_mean."
        ),
        "sources": {
            "positive_summary": str(args.positive_summary),
            "qwen_overall": str(args.qwen_overall),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
