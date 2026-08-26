#!/usr/bin/env python3
"""Freeze paired short/masked-short RankIC evidence for the PDF report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/"
    "single_stock_cninfo_v1/prompt_positive_v1/summary"
)
DEFAULT_OUTPUT = ROOT / "reports/comprehensive_research_report/audits/prompt_mask"
PROMPTS = {
    "profit": "盈利",
    "return": "收益",
    "excess_return": "超额收益",
    "loss": "亏损",
}
MODELS = {"roberta": "RoBERTa", "bge_m3": "BGE-M3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rankic(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["method"]: float(row["rankic"])
            for row in rows
            if row["method"] in PROMPTS
        }


def source_path(source: Path, model: str, representation: str, variant: str) -> Path:
    if representation == "prompt_mean":
        tag = "masked" if variant == "masked_short" else "short"
        name = f"prompt_late_fusion_{model}_{tag}_prompt_mean_yearly_overall.csv"
    else:
        name = f"prompt_late_fusion_{model}_{variant}_word_span_yearly_overall.csv"
    return source / name


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for model_key, model_name in MODELS.items():
        for representation in ("prompt_mean", "target_span"):
            short_path = source_path(args.source, model_key, representation, "short")
            masked_path = source_path(args.source, model_key, representation, "masked_short")
            short = read_rankic(short_path)
            masked = read_rankic(masked_path)
            if set(short) != set(PROMPTS) or set(masked) != set(PROMPTS):
                raise ValueError(f"incomplete prompt set in {short_path} or {masked_path}")
            for method, prompt in PROMPTS.items():
                rows.append(
                    {
                        "prompt": prompt,
                        "model": model_name,
                        "representation": representation,
                        "short_rankic": short[method],
                        "masked_short_rankic": masked[method],
                        "delta": masked[method] - short[method],
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "rankic_pairs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = {}
    for representation in ("prompt_mean", "target_span"):
        deltas = [float(row["delta"]) for row in rows if row["representation"] == representation]
        summaries[representation] = {
            "pairs": len(deltas),
            "wins": sum(delta > 0 for delta in deltas),
            "mean_delta": mean(deltas),
            "median_delta": median(deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
        }
    summary = {
        "scope": "Sina legacy aligned panel; 6+2+1; test years 2018-2026; next_day_return; Ridge",
        "treatment": "masked_short - short",
        "source_directory": str(args.source),
        "rows": rows,
        "summary": summaries,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
