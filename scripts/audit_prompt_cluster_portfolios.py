#!/usr/bin/env python3
"""Recompute execution-aware diagnostics for completed Sina prompt clusters.

The three experiment families use different portfolio protocols.  This audit
keeps them separate and reports daily basis points, position counts, turnover,
and costs from the lowest-level completed artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROMPTS = ("profit", "return", "excess_return", "loss")
PROMPT_ZH = {
    "profit": "盈利",
    "return": "收益",
    "excess_return": "超额收益",
    "loss": "亏损",
}
SPANS = {
    "profit": "profit_span",
    "return": "plain_return_span",
    "excess_return": "excess_span",
    "loss": "loss_span",
}
FACTOR_RE = re.compile(
    r"^soft_(excess_return|profit|return|loss)_(bge_m3|roberta)_"
    r"(masked_short|short)_(rank_ic|long_short)$"
)


def top20_execution(path: Path) -> dict[str, float | int]:
    frame = pd.read_parquet(
        path, columns=["stock_id", "entry_date", "actual_return", "prediction"]
    )
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    gross = 0.0
    cost = 0.0
    positive_days = 0
    holding_count = 0
    days = 0
    for _, yearly in frame.groupby(frame["entry_date"].dt.year):
        previous: dict[str, float] = {}
        for _, daily in yearly.groupby("entry_date", sort=True):
            daily = daily.sort_values(["prediction", "stock_id"], kind="mergesort")
            n = min(max(1, int(np.floor(len(daily) * 0.20))), len(daily) // 2)
            if n < 1:
                continue
            weights = {str(asset): 1.0 / n for asset in daily.tail(n)["stock_id"]}
            names = set(previous) | set(weights)
            bought = sum(max(weights.get(name, 0.0) - previous.get(name, 0.0), 0.0) for name in names)
            sold = sum(max(previous.get(name, 0.0) - weights.get(name, 0.0), 0.0) for name in names)
            outcomes = dict(
                zip(daily["stock_id"].astype(str), daily["actual_return"].astype(float))
            )
            daily_gross = sum(weight * outcomes.get(name, 0.0) for name, weight in weights.items())
            daily_cost = 0.0005 * bought + 0.0010 * sold
            gross += daily_gross
            cost += daily_cost
            positive_days += int(daily_gross - daily_cost > 0.0)
            holding_count += n
            days += 1
            previous = weights
        if previous:
            cost += 0.0010 * sum(max(weight, 0.0) for weight in previous.values())
    return {
        "signal_days": days,
        "average_holdings": holding_count / days,
        "gross_daily_bp": gross / days * 10_000,
        "cost_daily_bp": cost / days * 10_000,
        "net_daily_bp": (gross - cost) / days * 10_000,
        "positive_day_rate": positive_days / days,
    }


def audit_linear_and_hard(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for model in ("roberta", "bge_m3"):
        for variant in ("short", "masked_short"):
            for prompt in PROMPTS:
                representation = SPANS[prompt]
                paths = {
                    "linear_ridge": root
                    / "results"
                    / "regression"
                    / f"{prompt}_{model}_{variant}_{representation}"
                    / "stock_day_predictions.stock_day_predictions.parquet",
                    "hard_kmeans_ridge": root
                    / "results_cluster"
                    / "sina"
                    / f"{prompt}_{model}_{variant}_{representation}"
                    / "stock_day_predictions.parquet",
                }
                for method, path in paths.items():
                    rows.append(
                        {
                            "prompt": prompt,
                            "prompt_zh": PROMPT_ZH[prompt],
                            "model": model,
                            "variant": variant,
                            "method": method,
                            **top20_execution(path),
                        }
                    )
    return pd.DataFrame(rows)


def audit_soft(root: Path) -> pd.DataFrame:
    source = root / "soft_direction_tokens_v1" / "portfolio_fixed_gamma_0p10.csv"
    frame = pd.read_csv(source)
    frame = frame[(frame["cost"] == "china_a_5bp") & (frame["mode"] == "long_only")].copy()
    rows: list[dict] = []
    for row in frame.itertuples(index=False):
        match = FACTOR_RE.match(row.factor_id)
        if match is None:
            raise ValueError(f"unexpected factor id: {row.factor_id}")
        prompt, model, variant, objective = match.groups()
        daily = pd.read_parquet(Path(row.path).parent / "daily.parquet")
        rows.append(
            {
                "factor_id": row.factor_id,
                "prompt": prompt,
                "prompt_zh": PROMPT_ZH[prompt],
                "model": model,
                "variant": variant,
                "objective": objective,
                "days": len(daily),
                "gross_daily_bp": float(daily["gross_return"].mean() * 10_000),
                "cost_daily_bp": float(daily["transaction_cost"].mean() * 10_000),
                "net_daily_bp": float(daily["net_return"].mean() * 10_000),
                "average_holdings": float(daily["n_long"].mean()),
                "median_holdings": float(daily["n_long"].median()),
                "average_new_signals": float(daily["n_signal"].mean()),
                "average_raw_signals": float(daily["n_signal_raw"].mean()),
                "daily_turnover": float(daily["turnover"].mean()),
                "net_sharpe": float(row.net_sharpe),
                "net_geometric_annual_return": float(row.net_geometric_annual_return),
            }
        )
    return pd.DataFrame(rows)


def audit_umap_hdbscan(root: Path) -> pd.DataFrame:
    experiment = root / "results_umap_hdbscan" / "bge_m3_masked_short_loss_span"
    rows: list[dict] = []
    for year in range(2018, 2027):
        frame = pd.read_parquet(experiment / str(year) / "predictions.parquet")
        frame["entry_date"] = pd.to_datetime(frame["entry_date"])
        for method, column in (
            ("pca_ridge", "prediction_pca_ridge"),
            ("umap_hdbscan_ridge", "prediction_umap_hdbscan"),
        ):
            stock_day = frame.groupby(["stock_id", "entry_date"], as_index=False).agg(
                prediction=(column, "mean"), actual=("next_day_return", "mean")
            )
            observations: list[tuple[float, float, int]] = []
            for _, daily in stock_day.groupby("entry_date"):
                if len(daily) < 10:
                    continue
                n = max(1, int(np.ceil(len(daily) * 0.20)))
                daily = daily.sort_values("prediction")
                observations.append(
                    (
                        float(daily.tail(n)["actual"].mean()),
                        float(daily.head(n)["actual"].mean()),
                        n,
                    )
                )
            values = np.asarray(observations, dtype=float)
            rows.append(
                {
                    "year": year,
                    "method": method,
                    "valid_signal_days": len(values),
                    "average_long_holdings": float(values[:, 2].mean()),
                    "long_daily_bp": float(values[:, 0].mean() * 10_000),
                    "short_bucket_daily_bp": float(values[:, 1].mean() * 10_000),
                    "long_short_daily_bp": float((values[:, 0] - values[:, 1]).mean() * 10_000),
                }
            )
    return pd.DataFrame(rows)


def paired_mask_summary(frame: pd.DataFrame, metrics: list[str]) -> dict:
    index = [column for column in ("method", "prompt", "model", "objective") if column in frame]
    pivot = frame.pivot(index=index, columns="variant", values=metrics)
    output: dict[str, dict] = {}
    for metric in metrics:
        delta = pivot[(metric, "masked_short")] - pivot[(metric, "short")]
        output[metric] = {
            "mean_delta": float(delta.mean()),
            "wins": int((delta > 0).sum()),
            "pairs": int(delta.notna().sum()),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/comprehensive_research_report/audits/prompt_cluster_portfolios"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    linear_hard = audit_linear_and_hard(args.root)
    soft = audit_soft(args.root)
    hdbscan = audit_umap_hdbscan(args.root)
    linear_hard.to_csv(args.output_dir / "linear_hard_daily.csv", index=False)
    soft.to_csv(args.output_dir / "soft_long_only_daily.csv", index=False)
    hdbscan.to_csv(args.output_dir / "umap_hdbscan_daily.csv", index=False)

    hard_pivot = linear_hard.pivot(
        index=["prompt", "model", "variant"], columns="method", values="net_daily_bp"
    )
    hard_delta = hard_pivot["hard_kmeans_ridge"] - hard_pivot["linear_ridge"]
    hard_by_prompt = []
    for prompt, group in hard_pivot.groupby(level="prompt"):
        delta = group["hard_kmeans_ridge"] - group["linear_ridge"]
        hard_by_prompt.append(
            {
                "prompt": prompt,
                "prompt_zh": PROMPT_ZH[prompt],
                "linear_net_daily_bp": float(group["linear_ridge"].mean()),
                "hard_net_daily_bp": float(group["hard_kmeans_ridge"].mean()),
                "hard_minus_linear_net_daily_bp": float(delta.mean()),
                "hard_wins": int((delta > 0).sum()),
                "pairs": int(delta.notna().sum()),
            }
        )
    soft_leaders = (
        soft.sort_values(["prompt", "net_sharpe"], ascending=[True, False])
        .groupby("prompt", as_index=False)
        .first()
    )
    hdbscan_overall = hdbscan.drop(columns="year").groupby("method", as_index=False).mean(numeric_only=True)
    summary = {
        "protocols_are_not_level_comparable": True,
        "linear_hard": {
            "hard_minus_linear_net_daily_bp_mean": float(hard_delta.mean()),
            "hard_wins": int((hard_delta > 0).sum()),
            "pairs": int(hard_delta.notna().sum()),
            "method_means": linear_hard.groupby("method", as_index=False)
            .mean(numeric_only=True)
            .to_dict(orient="records"),
            "by_prompt": hard_by_prompt,
            "mask": paired_mask_summary(
                linear_hard, ["gross_daily_bp", "cost_daily_bp", "net_daily_bp"]
            ),
        },
        "soft_long_only": {
            "leaders_by_prompt": soft_leaders.to_dict(orient="records"),
            "mask": paired_mask_summary(
                soft, ["gross_daily_bp", "cost_daily_bp", "net_daily_bp", "net_sharpe"]
            ),
        },
        "umap_hdbscan": hdbscan_overall.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
