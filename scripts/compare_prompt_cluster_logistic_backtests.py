"""Compare prompt-token return clusters with Logistic baselines in one backtest.

All methods are restricted to the same stock-days and executable open-to-open
returns.  The script reports prediction diagnostics separately from portfolio
performance and never uses test returns to tune a model or portfolio setting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import (
    StrategyConfig,
    TransactionCostModel,
    backtest_ranked_strategy,
    strategy_metrics,
)


def _stock_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)


def load_cluster_scores(
    path: Path, *, feature_mode: str = "fusion", target: str = "next_day_return",
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"stock_id", "entry_date", "prediction", "actual_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"cluster predictions missing columns: {sorted(missing)}")
    if "feature_mode" in frame:
        frame = frame[frame["feature_mode"].eq(feature_mode)]
    if "target" in frame:
        frame = frame[frame["target"].eq(target)]
    result = frame[["stock_id", "entry_date", "prediction", "actual_return"]].copy()
    result["stock_id"] = _stock_id(result["stock_id"])
    result["entry_date"] = pd.to_datetime(result["entry_date"], errors="coerce")
    result = result.rename(columns={
        "prediction": "cluster_score", "actual_return": "label_return",
    }).dropna(subset=["stock_id", "entry_date", "cluster_score", "label_return"])
    if result.duplicated(["stock_id", "entry_date"]).any():
        raise ValueError("cluster predictions are not unique by stock-day")
    return result


def load_logistic_scores(
    path: Path, panel: pd.DataFrame, *, name: str,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"row_index", "entry_date", "probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Logistic predictions missing columns: {sorted(missing)}")
    keys = panel[["row_index", "stock_id"]].copy()
    if keys["row_index"].duplicated().any():
        raise ValueError("panel row_index must be unique")
    if "stock_id" not in frame:
        frame = frame.merge(keys, on="row_index", how="left", validate="many_to_one")
        if frame["stock_id"].isna().any():
            raise ValueError("Logistic predictions contain row_index values absent from panel")
    else:
        expected = frame[["row_index", "stock_id"]].merge(
            keys, on="row_index", how="left", suffixes=("", "_panel"),
            validate="many_to_one",
        )
        mismatch = _stock_id(expected["stock_id"]).ne(
            _stock_id(expected["stock_id_panel"])
        )
        if mismatch.any():
            raise ValueError("Logistic stock_id values disagree with panel row_index")
    frame["stock_id"] = _stock_id(frame["stock_id"])
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    score_column = f"logistic_{name}"
    return frame.dropna(subset=["stock_id", "entry_date", "probability"]).groupby(
        ["stock_id", "entry_date"], as_index=False, observed=True,
    ).agg(**{score_column: ("probability", "mean")})


def parse_logistic_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--logistic must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip().replace("-", "_")
    if not name or not name.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError("Logistic NAME must be alphanumeric/underscore")
    return name, Path(raw_path)


def build_common_signals(
    cluster: pd.DataFrame,
    logistics: list[tuple[str, pd.DataFrame]],
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    common = cluster.copy()
    for _, frame in logistics:
        common = common.merge(
            frame, on=["stock_id", "entry_date"], how="inner", validate="one_to_one"
        )
    market_columns = [
        "stock_id", "entry_date", "open_to_open_return", "can_buy", "can_sell",
        "eligible_signal",
    ]
    missing = set(market_columns).difference(market.columns)
    if missing:
        raise ValueError(f"market data missing columns: {sorted(missing)}")
    executable = market[market_columns].copy()
    executable["stock_id"] = _stock_id(executable["stock_id"])
    executable["entry_date"] = pd.to_datetime(executable["entry_date"], errors="coerce")
    if executable.duplicated(["stock_id", "entry_date"]).any():
        raise ValueError("market data are not unique by stock-day")
    common = common.merge(
        executable, on=["stock_id", "entry_date"], how="left", validate="one_to_one"
    )
    finite_market = np.isfinite(pd.to_numeric(common["open_to_open_return"], errors="coerce"))
    eligible = common["eligible_signal"].fillna(False).astype(bool)
    audit = {
        "prediction_common_stock_days": int(len(common)),
        "finite_open_to_open_stock_days": int(finite_market.sum()),
        "eligible_executable_stock_days": int((finite_market & eligible).sum()),
    }
    common = common[finite_market & eligible].copy()

    common["score_cluster"] = common["cluster_score"]
    methods = ["cluster"]
    for name, _ in logistics:
        raw = f"logistic_{name}"
        common[f"score_{name}"] = common[raw]
        cluster_rank = common.groupby("entry_date", observed=True)["cluster_score"].rank(
            method="average", pct=True
        )
        logistic_rank = common.groupby("entry_date", observed=True)[raw].rank(
            method="average", pct=True
        )
        common[f"score_blend_{name}"] = (cluster_rank + logistic_rank) / 2.0
        methods.extend([name, f"blend_{name}"])
    audit["methods"] = methods
    return common, audit


def prediction_diagnostics(frame: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    actual_direction = frame["label_return"].gt(0).to_numpy()
    for method in methods:
        scores = frame[f"score_{method}"].to_numpy(dtype=float)
        if method.startswith("cluster"):
            predicted_direction = scores > 0.0
        elif method.startswith("blend"):
            predicted_direction = scores > 0.5
        else:
            predicted_direction = scores > 0.5
        daily_ics = []
        for _, part in frame.groupby("entry_date", observed=True):
            if len(part) < 5 or part[f"score_{method}"].nunique() < 2:
                continue
            value = spearmanr(
                part[f"score_{method}"], part["open_to_open_return"]
            ).statistic
            if np.isfinite(value):
                daily_ics.append(float(value))
        rows.append({
            "method": method,
            "stock_days": int(len(frame)),
            "dates": int(frame["entry_date"].nunique()),
            "label_direction_accuracy": float(np.mean(predicted_direction == actual_direction)),
            "execution_return_daily_ic": float(np.mean(daily_ics)) if daily_ics else np.nan,
            "execution_return_global_ic": float(spearmanr(
                scores, frame["open_to_open_return"].to_numpy(dtype=float)
            ).statistic),
        })
    return pd.DataFrame(rows)


def _calendarized(daily: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    result = daily.set_index("date").reindex(calendar).rename_axis("date").reset_index()
    numeric = [column for column in result.columns if column != "date"]
    result[numeric] = result[numeric].fillna(0.0)
    result["cumulative_gross_return"] = (1.0 + result["gross_return"]).cumprod() - 1.0
    result["cumulative_net_return"] = (1.0 + result["net_return"]).cumprod() - 1.0
    return result


def block_bootstrap_difference(
    cluster_returns: np.ndarray,
    baseline_returns: np.ndarray,
    *,
    samples: int = 2000,
    block_length: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    cluster_values = np.asarray(cluster_returns, dtype=float)
    baseline_values = np.asarray(baseline_returns, dtype=float)
    if cluster_values.shape != baseline_values.shape or cluster_values.ndim != 1:
        raise ValueError("paired return series must be aligned one-dimensional arrays")
    differences = cluster_values - baseline_values
    n = len(differences)
    if n == 0:
        raise ValueError("paired return series are empty")
    length = min(max(1, block_length), n)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for draw in range(samples):
        pieces = []
        while sum(len(piece) for piece in pieces) < n:
            start = int(rng.integers(0, n - length + 1))
            pieces.append(differences[start : start + length])
        means[draw] = np.concatenate(pieces)[:n].mean()
    return {
        "mean_daily_difference": float(differences.mean()),
        "annualized_mean_difference": float(differences.mean() * 252.0),
        "ci_low_annualized": float(np.quantile(means, 0.025) * 252.0),
        "ci_high_annualized": float(np.quantile(means, 0.975) * 252.0),
        "bootstrap_probability_positive": float(np.mean(means > 0.0)),
    }


def _flatten_metrics(
    *, method: str, portfolio: str, quantiles: int, metrics: dict[str, object],
) -> dict[str, object]:
    gross = metrics["gross"]
    net = metrics["net"]
    execution = metrics["execution"]
    return {
        "method": method, "portfolio": portfolio, "quantiles": quantiles,
        "n_days": int(net["n_days"]),
        "gross_annualized_return": gross["geometric_annualized_return"],
        "gross_sharpe": gross["sharpe"],
        "net_annualized_return": net["geometric_annualized_return"],
        "net_sharpe": net["sharpe"],
        "net_cumulative_return": net["cumulative_return"],
        "net_max_drawdown": net["max_drawdown"],
        "net_positive_day_rate": net["positive_day_rate"],
        "mean_daily_turnover": execution["mean_daily_turnover"],
        "annualized_turnover": execution["annualized_turnover"],
        "annualized_transaction_cost": execution["annualized_transaction_cost"],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel, columns=["row_index", "stock_id"])
    cluster = load_cluster_scores(
        args.cluster_predictions,
        feature_mode=args.cluster_feature_mode,
        target=args.cluster_target,
    )
    logistic_frames = [
        (name, load_logistic_scores(path, panel, name=name))
        for name, path in args.logistic
    ]
    market = pd.read_parquet(args.market)
    common, audit = build_common_signals(cluster, logistic_frames, market)
    methods = list(audit["methods"])
    diagnostics = prediction_diagnostics(common, methods)
    diagnostics.to_csv(args.output_dir / "prediction_diagnostics.csv", index=False)
    common.to_parquet(args.output_dir / "common_executable_signals.parquet", index=False)

    market = market.copy()
    market["stock_id"] = _stock_id(market["stock_id"])
    market["entry_date"] = pd.to_datetime(market["entry_date"], errors="coerce")
    relevant_assets = set(common["stock_id"])
    start, last_signal = common["entry_date"].min(), common["entry_date"].max()
    later = market.loc[market["entry_date"].gt(last_signal), "entry_date"]
    end = later.min() if not later.empty else last_signal
    market = market[
        market["stock_id"].isin(relevant_assets)
        & market["entry_date"].between(start, end)
    ].copy()
    calendar = pd.DatetimeIndex(sorted(market["entry_date"].dropna().unique()))
    cost_model = TransactionCostModel(
        name="custom", commission_bps=args.one_way_cost_bps,
        stamp_duty_bps=0.0, slippage_bps=0.0,
    )
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    daily_lookup: dict[tuple[str, str, int], pd.DataFrame] = {}
    for quantiles in args.quantiles:
        for long_only in (False, True):
            portfolio = "long_only" if long_only else "long_short"
            for method in methods:
                signal = common[["stock_id", "entry_date", f"score_{method}"]].rename(
                    columns={f"score_{method}": "prediction"}
                )
                result = backtest_ranked_strategy(
                    signal,
                    prediction="prediction",
                    date="entry_date",
                    asset="stock_id",
                    market_data=market,
                    market_return="open_to_open_return",
                    can_buy="can_buy",
                    can_sell="can_sell",
                    config=StrategyConfig(
                        quantiles=quantiles,
                        min_stocks_per_day=max(args.min_stocks_per_day, quantiles),
                        long_only=long_only,
                        ewct_gamma=1.0,
                        no_signal_action="close",
                        long_short_normalization="gross",
                    ),
                    costs=cost_model,
                )
                daily = _calendarized(result.daily, calendar)
                key = (method, portfolio, quantiles)
                daily_lookup[key] = daily
                run_dir = args.output_dir / f"q{quantiles}" / portfolio / method
                run_dir.mkdir(parents=True, exist_ok=True)
                daily.to_parquet(run_dir / "daily.parquet", index=False)
                result.holdings.to_parquet(run_dir / "holdings.parquet", index=False)
                result.trades.to_parquet(run_dir / "trades.parquet", index=False)
                summary_rows.append(_flatten_metrics(
                    method=method, portfolio=portfolio, quantiles=quantiles,
                    metrics=strategy_metrics(daily),
                ))
                for year, part in daily.groupby(daily["date"].dt.year, observed=True):
                    yearly_rows.append(_flatten_metrics(
                        method=method, portfolio=portfolio, quantiles=quantiles,
                        metrics=strategy_metrics(part),
                    ) | {"year": int(year)})

    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    summary.to_csv(args.output_dir / "portfolio_summary.csv", index=False)
    yearly.to_csv(args.output_dir / "portfolio_yearly.csv", index=False)

    comparisons = []
    baselines = [name for name, _ in logistic_frames]
    for quantiles in args.quantiles:
        for portfolio in ("long_short", "long_only"):
            cluster_daily = daily_lookup[("cluster", portfolio, quantiles)]
            for baseline in baselines:
                base_daily = daily_lookup[(baseline, portfolio, quantiles)]
                result = block_bootstrap_difference(
                    cluster_daily["net_return"].to_numpy(),
                    base_daily["net_return"].to_numpy(),
                    samples=args.bootstrap_samples,
                    block_length=args.bootstrap_block_length,
                )
                comparisons.append({
                    "quantiles": quantiles, "portfolio": portfolio,
                    "method": "cluster", "baseline": baseline, **result,
                })
    paired = pd.DataFrame(comparisons)
    paired.to_csv(args.output_dir / "paired_cluster_vs_logistic.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        for portfolio in ("long_short", "long_only"):
            quantiles = max(args.quantiles)
            figure, axis = plt.subplots(figsize=(10, 5.5))
            for method in methods:
                daily = daily_lookup[(method, portfolio, quantiles)]
                axis.plot(
                    daily["date"], 1.0 + daily["cumulative_net_return"],
                    label=method, linewidth=1.3,
                )
            axis.axhline(1.0, color="black", linewidth=0.7)
            axis.set_title(f"Net wealth: q={quantiles}, {portfolio}, 5bp each trade")
            axis.set_ylabel("Wealth")
            axis.grid(alpha=0.2)
            axis.legend()
            figure.tight_layout()
            figure.savefig(args.output_dir / f"net_wealth_q{quantiles}_{portfolio}.png", dpi=160)
            plt.close(figure)
    except ImportError:
        audit["plot_warning"] = "matplotlib unavailable"

    audit.update({
        "format_version": "prompt_cluster_logistic_backtest_comparison_v1",
        "cluster_predictions": str(args.cluster_predictions),
        "logistic_predictions": {name: str(path) for name, path in args.logistic},
        "market": str(args.market),
        "market_return": "open_to_open_return",
        "one_way_cost_bps": args.one_way_cost_bps,
        "holding_rule": "one open-to-open period, then close",
        "long_short_normalization": "gross exposure",
        "shorting_note": "long-short is an academic portfolio; shortability data unavailable",
        "calendar_days": int(len(calendar)),
    })
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-predictions", type=Path, required=True)
    parser.add_argument("--cluster-feature-mode", default="fusion")
    parser.add_argument("--cluster-target", default="next_day_return")
    parser.add_argument("--logistic", type=parse_logistic_spec, action="append", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantiles", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--min-stocks-per-day", type=int, default=5)
    parser.add_argument("--one-way-cost-bps", type=float, default=5.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-block-length", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if len({name for name, _ in args.logistic}) != len(args.logistic):
        raise ValueError("Logistic names must be unique")
    if any(value < 2 for value in args.quantiles):
        raise ValueError("quantiles must be at least two")
    if args.one_way_cost_bps < 0:
        raise ValueError("one-way cost must be non-negative")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
