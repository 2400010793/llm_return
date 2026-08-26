"""Aggregate soft-cluster family prediction and long/short leg results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.artifacts import atomic_json


def _bootstrap(values: np.ndarray, *, samples: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        rng.choice(values, len(values), replace=True).mean() for _ in range(samples)
    ])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    files = sorted(args.fold_root.glob("*/*/*/metrics.json"))
    if args.strict and len(files) != 54:
        raise ValueError(f"expected 54 soft-family folds; found {len(files)}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    yearly_rows = []
    predictions = []
    for path, report in zip(files, reports):
        yearly_rows.append({
            "feature_mode": report["feature_mode"], "target": report["target"],
            "test_year": report["test_year"], **report["selected"],
            **report["test_metrics"],
        })
        values = pd.read_parquet(path.parent / "stock_day_predictions.parquet")
        values["feature_mode"] = report["feature_mode"]
        values["target"] = report["target"]
        values["test_year"] = report["test_year"]
        predictions.append(values)
    yearly = pd.DataFrame(yearly_rows)
    all_predictions = pd.concat(predictions, ignore_index=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(args.output_root / "yearly_metrics.csv", index=False)
    all_predictions.to_parquet(args.output_root / "stock_day_predictions_all.parquet", index=False)

    family_rows = []
    overall_rows = []
    round_trip_cost = 0.001
    for keys, part in all_predictions.groupby(["feature_mode", "target"]):
        family_values = {}
        for family, selected in part.groupby("family"):
            returns = selected["actual_return"].to_numpy(dtype=float)
            low, high = _bootstrap(returns)
            family_values[family] = returns
            family_rows.append({
                "feature_mode": keys[0], "target": keys[1], "family": family,
                "stock_days": len(returns), "mean_return": float(returns.mean()),
                "median_return": float(np.median(returns)),
                "positive_rate": float((returns > 0).mean()),
                "ci_low": low, "ci_high": high,
            })
        negative = family_values["negative"]
        positive = family_values["positive"]
        rng = np.random.default_rng(42)
        spread_draws = np.asarray([
            rng.choice(positive, len(positive), replace=True).mean()
            - rng.choice(negative, len(negative), replace=True).mean()
            for _ in range(2000)
        ])
        selected_yearly = yearly[
            (yearly["feature_mode"] == keys[0]) & (yearly["target"] == keys[1])
        ]
        long_gross = float(positive.mean())
        short_gross = float(-negative.mean())
        overall_rows.append({
            "feature_mode": keys[0], "target": keys[1],
            "years": int(selected_yearly["test_year"].nunique()),
            "stock_days": int(len(part)),
            "mean_daily_ic": float(selected_yearly["daily_ic"].mean()),
            "positive_ic_years": int((selected_yearly["daily_ic"] > 0).sum()),
            "direction_accuracy": float(np.average(
                selected_yearly["direction_accuracy"],
                weights=selected_yearly["stock_days"],
            )),
            "modal_k": int(selected_yearly["k"].mode().iloc[0]),
            "mean_ari": float(selected_yearly["ari_mean"].mean()),
            "long_gross": long_gross, "long_net": long_gross - round_trip_cost,
            "short_gross": short_gross, "short_net": short_gross - round_trip_cost,
            "long_short_gross_on_gross": (long_gross + short_gross) / 2,
            "long_short_net_on_gross": (long_gross + short_gross) / 2 - round_trip_cost,
            "long_short_spread": long_gross + short_gross,
            "spread_ci_low": float(np.quantile(spread_draws, 0.025)),
            "spread_ci_high": float(np.quantile(spread_draws, 0.975)),
        })
    family = pd.DataFrame(family_rows)
    overall = pd.DataFrame(overall_rows)
    family.to_csv(args.output_root / "family_returns.csv", index=False)
    overall.to_csv(args.output_root / "overall.csv", index=False)
    lines = [
        "# BGE-M3 收益Token软聚类族预测",
        "",
        f"- 完成：{len(reports)}/54 folds。",
        "- k=2/4/6/8、收缩强度和软分配温度仅由2年验证期选择；测试年不参与选择。",
        "- 成本为单边5bp，即每个多头或空头信号往返10bp。3日结果是信号级事件收益，尚未处理重叠持仓。",
        "",
        "## 总体",
        "",
    ]
    shown = overall.copy()
    for column in (
        "direction_accuracy", "long_gross", "long_net", "short_gross", "short_net",
        "long_short_gross_on_gross", "long_short_net_on_gross", "long_short_spread",
        "spread_ci_low", "spread_ci_high",
    ):
        shown[column] = shown[column].map(lambda value: f"{value:+.4%}")
    lines.extend(shown.to_markdown(index=False).splitlines())
    lines.extend(["", "## 三族收益", ""])
    family_shown = family.copy()
    for column in ("mean_return", "median_return", "positive_rate", "ci_low", "ci_high"):
        family_shown[column] = family_shown[column].map(lambda value: f"{value:+.4%}")
    lines.extend(family_shown.to_markdown(index=False).splitlines())
    (args.output_root / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    audit = {
        "format_version": "soft_return_family_summary_v1",
        "folds": len(reports), "expected_folds": 54,
        "complete": len(reports) == 54,
    }
    atomic_json(args.output_root / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
