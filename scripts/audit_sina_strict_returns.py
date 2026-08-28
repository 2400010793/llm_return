"""Audit Sina return results without selecting models on test labels.

The completed Sina experiments contain final-test predictions for 20 prompt or
mask variants, 20 variant-gate models, and 20 Ridge regressions.  This script
does not rank those reports by test accuracy, Rank IC, or portfolio return.
Instead, it selects one candidate per test year using only the validation
metrics persisted in each report, reconstructs stock-day predictions, and
forms equal-weighted portfolios.

The source panel is sparse in the cross-section.  Therefore the audit reports
event-day returns and calendar coverage, rather than mechanically annualizing
the 32 or fewer usable dates.  Transaction-cost numbers are explicitly
illustrative scenarios because the current artifacts do not contain holdings,
turnover, prices, or a complete daily market panel.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_DATA_ROOT = Path("/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812")
DEFAULT_MIN_STOCKS = (5, 10, 20)
DEFAULT_COST_BPS = (0.0, 10.0, 25.0, 50.0)
DEFAULT_RANK_IC_THRESHOLDS = (1, 3)
SELECTION_VERSION = "sina_strict_validation_selection_v1"


def _finite_float(value: Any) -> float:
    """Convert a metric to a float, returning NaN for missing values."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _parse_ints(value: str, *, name: str, minimum: int = 1) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or any(item < minimum for item in values):
        raise ValueError(f"{name} must contain integers >= {minimum}")
    return values


def _parse_floats(value: str, *, name: str, minimum: float = 0.0) -> tuple[float, ...]:
    values = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or any(item < minimum for item in values):
        raise ValueError(f"{name} must contain numbers >= {minimum}")
    return values


def load_panel(path: Path) -> pd.DataFrame:
    """Load and validate the canonical article-to-stock-day panel."""
    panel = pd.read_parquet(path).copy()
    required = {"article_id", "stock_id", "entry_date", "next_day_return"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {', '.join(sorted(missing))}")
    if panel["article_id"].isna().any() or panel["article_id"].duplicated().any():
        raise ValueError("panel article_id must be non-null and unique")
    panel["article_id"] = panel["article_id"].astype(str)
    panel["entry_date"] = pd.to_datetime(panel["entry_date"], errors="coerce")
    panel["next_day_return"] = pd.to_numeric(panel["next_day_return"], errors="coerce")
    invalid_labeled_date = panel["entry_date"].isna() & panel["next_day_return"].notna()
    if invalid_labeled_date.any():
        raise ValueError("panel contains labeled rows with invalid entry_date values")
    return panel[["article_id", "stock_id", "entry_date", "next_day_return"]]


def _report_fold(report: Mapping[str, Any], test_year: int) -> dict[str, Any]:
    folds = {
        int(row["test_year"]): row
        for row in report.get("results", [])
        if row.get("test_year") is not None
    }
    if test_year not in folds:
        raise ValueError(f"report has no test fold for {test_year}")
    return folds[test_year]


def _validation_metrics(fold: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = fold.get("validation_metrics")
    if isinstance(metrics, Mapping):
        return metrics
    return {}


def _candidate_name(path: Path) -> str:
    return path.stem


def load_reports(root: Path, *, expected_count: int = 20) -> list[dict[str, Any]]:
    """Load one complete result family and retain its source path."""
    paths = sorted(root.glob("*.json"))
    if len(paths) != expected_count:
        raise ValueError(
            f"expected {expected_count} reports in {root}, found {len(paths)}"
        )
    reports = []
    for path in paths:
        report = json.loads(path.read_text())
        report["_path"] = str(path)
        report["_name"] = _candidate_name(path)
        reports.append(report)
    return reports


def _candidate_score(kind: str, fold: Mapping[str, Any]) -> dict[str, float]:
    """Extract validation-only fields used by the pre-registered selector."""
    metrics = _validation_metrics(fold)
    if kind in {"classification", "gate"}:
        return {
            "primary": _finite_float(
                metrics.get("accuracy", fold.get("validation_accuracy"))
            ),
            "secondary": _finite_float(metrics.get("balanced_accuracy")),
            "tertiary": _finite_float(metrics.get("auc")),
            "validation_n": _finite_float(metrics.get("n")),
            "rank_ic_days": float("nan"),
        }
    return {
        "primary": _finite_float(metrics.get("rank_ic_mean")),
        "secondary": _finite_float(metrics.get("oos_r2_vs_historical_mean")),
        "tertiary": -_finite_float(metrics.get("mse")),
        "validation_n": _finite_float(metrics.get("n")),
        "rank_ic_days": _finite_float(metrics.get("rank_ic_days")),
    }


def select_per_test_year(
    reports: Iterable[Mapping[str, Any]],
    *,
    kind: str,
    min_validation_n: int = 100,
    min_rank_ic_days: int | None = None,
) -> list[dict[str, Any]]:
    """Select candidates using only persisted validation metrics.

    The returned records intentionally contain no test metric.  Regression
    selection requires a minimum number of validation cross-sections when a
    Rank IC rule is used; if no report passes that requirement, the test year
    remains unselected instead of silently falling back to test performance.
    """
    report_list = list(reports)
    years = sorted({
        int(row["test_year"])
        for report in report_list
        for row in report.get("results", [])
        if row.get("test_year") is not None
    })
    output: list[dict[str, Any]] = []
    for year in years:
        candidates: list[dict[str, Any]] = []
        for report in report_list:
            fold = _report_fold(report, year)
            score = _candidate_score(kind, fold)
            eligible = (
                np.isfinite(score["primary"])
                and np.isfinite(score["validation_n"])
                and score["validation_n"] >= min_validation_n
            )
            if kind == "regression" and min_rank_ic_days is not None:
                eligible = eligible and (
                    np.isfinite(score["rank_ic_days"])
                    and score["rank_ic_days"] >= min_rank_ic_days
                )
            candidates.append({
                "model": report["_name"],
                "report": report["_path"],
                "test_year": year,
                "eligible": bool(eligible),
                **score,
            })

        eligible = [row for row in candidates if row["eligible"]]
        eligible.sort(key=lambda row: (
            -row["primary"],
            -row["secondary"] if np.isfinite(row["secondary"]) else np.inf,
            -row["tertiary"] if np.isfinite(row["tertiary"]) else np.inf,
            row["model"],
        ))
        leaderboard = sorted(
            candidates,
            key=lambda row: (
                not row["eligible"],
                -row["primary"] if np.isfinite(row["primary"]) else np.inf,
                row["model"],
            ),
        )
        selected = eligible[0] if eligible else None
        output.append({
            "test_year": year,
            "status": "selected" if selected else "insufficient_validation_data",
            "selected_model": selected["model"] if selected else None,
            "selected_report": selected["report"] if selected else None,
            "eligible_count": len(eligible),
            "candidate_count": len(candidates),
            "leaderboard": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"report", "eligible"}
                }
                for row in leaderboard
            ],
        })
    return output


def _canonical_classification_predictions(
    predictions: pd.DataFrame, panel: pd.DataFrame, *, test_year: int
) -> pd.DataFrame:
    required = {"article_id", "probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"classification predictions missing {sorted(missing)}")
    values = predictions.copy()
    values["article_id"] = values["article_id"].astype(str)
    if values["article_id"].duplicated().any():
        raise ValueError("classification prediction article_id is not unique")
    canonical = panel.merge(
        values[["article_id", "probability"]],
        on="article_id",
        how="inner",
        validate="one_to_one",
    )
    if len(canonical) != len(values):
        raise ValueError("classification predictions do not align one-to-one to panel")
    canonical = canonical.rename(columns={
        "probability": "score",
        "next_day_return": "actual_return",
    })
    canonical["test_year"] = int(test_year)
    return canonical[[
        "article_id", "stock_id", "entry_date", "actual_return", "score", "test_year"
    ]]


def _canonical_regression_predictions(
    predictions: pd.DataFrame, *, test_year: int
) -> pd.DataFrame:
    required = {"stock_id", "entry_date", "actual_return", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"regression predictions missing {sorted(missing)}")
    values = predictions.copy().rename(columns={"prediction": "score"})
    values["entry_date"] = pd.to_datetime(values["entry_date"], errors="coerce")
    values["actual_return"] = pd.to_numeric(values["actual_return"], errors="coerce")
    values["score"] = pd.to_numeric(values["score"], errors="coerce")
    values["test_year"] = int(test_year)
    return values[["stock_id", "entry_date", "actual_return", "score", "test_year"]]


def load_candidate_predictions(
    report: Mapping[str, Any],
    *,
    kind: str,
    test_year: int,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Read only the selected candidate's persisted test predictions."""
    key = "stock_day_predictions" if kind == "regression" else "predictions"
    path = report.get(key)
    if not path:
        raise ValueError(f"report {report.get('_name')} has no {key} path")
    predictions = pd.read_parquet(path)
    if "test_year" in predictions:
        predictions = predictions[predictions["test_year"] == test_year].copy()
    if kind == "regression":
        return _canonical_regression_predictions(predictions, test_year=test_year)
    return _canonical_classification_predictions(
        predictions, panel, test_year=test_year
    )


def aggregate_stock_day(values: pd.DataFrame) -> pd.DataFrame:
    """Average document scores and reject inconsistent stock-day labels."""
    required = {"stock_id", "entry_date", "actual_return", "score", "test_year"}
    missing = required.difference(values.columns)
    if missing:
        raise ValueError(f"stock-day aggregation missing {sorted(missing)}")
    frame = values.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["actual_return"] = pd.to_numeric(frame["actual_return"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["stock_id", "entry_date", "actual_return", "score"])
    groups = frame.groupby(["stock_id", "entry_date"], sort=True, observed=True)
    inconsistent = groups["actual_return"].nunique(dropna=False).gt(1)
    if inconsistent.any():
        raise ValueError("inconsistent actual returns within stock-day groups")
    return groups.agg(
        actual_return=("actual_return", "first"),
        score=("score", "mean"),
        n_predictions=("score", "size"),
        test_year=("test_year", "first"),
    ).reset_index()


def form_portfolio(stock_day: pd.DataFrame, *, min_stocks: int) -> pd.DataFrame:
    """Form five equal-weighted portfolios on dates with enough stocks."""
    rows: list[dict[str, Any]] = []
    for day, group in stock_day.groupby("entry_date", sort=True):
        group = group.dropna(subset=["actual_return", "score"])
        if len(group) < min_stocks or len(group) < 5:
            continue
        buckets = pd.qcut(
            group["score"].rank(method="first"), q=5, labels=False, duplicates="drop"
        )
        unique_buckets = sorted(buckets.dropna().unique())
        if len(unique_buckets) != 5:
            continue
        returns = {
            f"q{int(bucket) + 1}": float(group.loc[buckets == bucket, "actual_return"].mean())
            for bucket in unique_buckets
        }
        rows.append({
            "date": day,
            "year": int(pd.Timestamp(day).year),
            **returns,
            "market": float(group["actual_return"].mean()),
            "long_short": returns["q5"] - returns["q1"],
            "n": int(len(group)),
        })
    columns = ["date", "year", "q1", "q2", "q3", "q4", "q5", "market", "long_short", "n"]
    return pd.DataFrame(rows, columns=columns)


def _sign_test_pvalue(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > 0]
    if len(nonzero) == 0:
        return 1.0
    positives = int(np.sum(nonzero > 0))
    n = len(nonzero)
    return float(sum(math.comb(n, index) for index in range(positives, n + 1)) / 2**n)


def _bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[float | None]:
    if len(values) == 0 or n_bootstrap < 1:
        return [None, None]
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    samples = values[indices]
    if statistic == "mean":
        draws = samples.mean(axis=1)
    elif statistic == "cumulative_return":
        with np.errstate(over="ignore", invalid="ignore"):
            draws = np.prod(1.0 + samples, axis=1) - 1.0
    else:
        raise ValueError(f"unsupported bootstrap statistic: {statistic}")
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return [None, None]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize_series(
    values: Iterable[float],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Summarize event-day returns without calendar annualization."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "n_days": 0,
            "mean_event_day_return": None,
            "cumulative_return": None,
            "positive_day_rate": None,
            "sign_test_p_greater_0": None,
            "bootstrap_95_ci": {
                "mean_event_day_return": [None, None],
                "cumulative_return": [None, None],
            },
        }
    wealth = np.cumprod(1.0 + array)
    cumulative = float(wealth[-1] - 1.0) if np.all(1.0 + array > 0) else None
    return {
        "n_days": int(len(array)),
        "mean_event_day_return": float(array.mean()),
        "cumulative_return": cumulative,
        "positive_day_rate": float(np.mean(array > 0)),
        "sign_test_p_greater_0": _sign_test_pvalue(array),
        "bootstrap_95_ci": {
            "mean_event_day_return": _bootstrap_ci(
                array, statistic="mean", rng=rng, n_bootstrap=n_bootstrap
            ),
            "cumulative_return": _bootstrap_ci(
                array, statistic="cumulative_return", rng=rng, n_bootstrap=n_bootstrap
            ),
        },
    }


def cost_sensitivity(
    portfolio: pd.DataFrame,
    *,
    return_column: str,
    assumed_turnover: float,
    cost_bps: Iterable[float],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Apply illustrative constant cost drags to an event-day return series."""
    gross = pd.to_numeric(portfolio[return_column], errors="coerce").to_numpy(float)
    gross = gross[np.isfinite(gross)]
    result: dict[str, Any] = {
        "return_column": return_column,
        "assumed_turnover_per_event_day": float(assumed_turnover),
        "cost_model": "constant_cost_bps_times_assumed_turnover; not holdings-level turnover",
        "scenarios": [],
    }
    for bps in cost_bps:
        net = gross - float(bps) / 10000.0 * assumed_turnover
        result["scenarios"].append({
            "cost_bps": float(bps),
            "daily_cost_drag": float(bps) / 10000.0 * assumed_turnover,
            "net": summarize_series(net, rng=rng, n_bootstrap=n_bootstrap),
        })
    if len(gross) and assumed_turnover > 0:
        result["break_even_cost_bps_per_turnover_unit"] = float(
            gross.mean() * 10000.0 / assumed_turnover
        )
    else:
        result["break_even_cost_bps_per_turnover_unit"] = None
    return result


def portfolio_summary(
    portfolio: pd.DataFrame,
    *,
    cost_bps: Iterable[float],
    long_only_turnover: float,
    long_short_turnover: float,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Summarize gross portfolios, bootstrap intervals, and cost scenarios."""
    summary = {
        "effective_dates": int(len(portfolio)),
        "date_min": str(portfolio["date"].min().date()) if len(portfolio) else None,
        "date_max": str(portfolio["date"].max().date()) if len(portfolio) else None,
        "mean_stocks_per_date": float(portfolio["n"].mean()) if len(portfolio) else None,
        "by_year": {},
        "gross": {
            "market_equal_weight": summarize_series(
                portfolio.get("market", pd.Series(dtype=float)),
                rng=rng,
                n_bootstrap=n_bootstrap,
            ),
            "q1_low_score": summarize_series(
                portfolio.get("q1", pd.Series(dtype=float)),
                rng=rng,
                n_bootstrap=n_bootstrap,
            ),
            "q5_high_score_long_only": summarize_series(
                portfolio.get("q5", pd.Series(dtype=float)),
                rng=rng,
                n_bootstrap=n_bootstrap,
            ),
            "q5_minus_q1_long_short": summarize_series(
                portfolio.get("long_short", pd.Series(dtype=float)),
                rng=rng,
                n_bootstrap=n_bootstrap,
            ),
        },
        "cost_sensitivity": {},
    }
    for year, year_portfolio in portfolio.groupby("year", sort=True):
        summary["by_year"][str(int(year))] = {
            "effective_dates": int(len(year_portfolio)),
            "mean_stocks_per_date": float(year_portfolio["n"].mean()),
            "gross": {
                "market_equal_weight": summarize_series(
                    year_portfolio["market"],
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                ),
                "q5_high_score_long_only": summarize_series(
                    year_portfolio["q5"],
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                ),
                "q5_minus_q1_long_short": summarize_series(
                    year_portfolio["long_short"],
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                ),
            },
        }
    summary["cost_sensitivity"]["q5_high_score_long_only"] = cost_sensitivity(
        portfolio,
        return_column="q5",
        assumed_turnover=long_only_turnover,
        cost_bps=cost_bps,
        rng=rng,
        n_bootstrap=n_bootstrap,
    )
    summary["cost_sensitivity"]["q5_minus_q1_long_short"] = cost_sensitivity(
        portfolio,
        return_column="long_short",
        assumed_turnover=long_short_turnover,
        cost_bps=cost_bps,
        rng=rng,
        n_bootstrap=n_bootstrap,
    )
    return summary


def _selected_stock_days(
    reports: Iterable[Mapping[str, Any]],
    selection: Iterable[Mapping[str, Any]],
    *,
    kind: str,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    by_name = {report["_name"]: report for report in reports}
    frames: list[pd.DataFrame] = []
    for row in selection:
        if row["status"] != "selected":
            continue
        report = by_name[row["selected_model"]]
        values = load_candidate_predictions(
            report, kind=kind, test_year=int(row["test_year"]), panel=panel
        )
        values["selected_model"] = row["selected_model"]
        frames.append(values)
    if not frames:
        return pd.DataFrame(
            columns=["stock_id", "entry_date", "actual_return", "score", "test_year"]
        )
    return aggregate_stock_day(pd.concat(frames, ignore_index=True))


def audit_family(
    reports: list[dict[str, Any]],
    *,
    kind: str,
    panel: pd.DataFrame,
    min_stocks: tuple[int, ...],
    cost_bps: tuple[float, ...],
    long_only_turnover: float,
    long_short_turnover: float,
    n_bootstrap: int,
    seed: int,
    rank_ic_thresholds: tuple[int, ...] = DEFAULT_RANK_IC_THRESHOLDS,
    primary_rank_ic_days: int = 3,
    min_validation_n: int = 100,
) -> dict[str, Any]:
    """Audit one result family under validation-only selection."""
    if kind == "regression":
        thresholds = tuple(sorted(set(rank_ic_thresholds), reverse=True))
        policies = {
            f"validation_rank_ic_min_days_{threshold}": select_per_test_year(
                reports,
                kind=kind,
                min_validation_n=min_validation_n,
                min_rank_ic_days=threshold,
            )
            for threshold in thresholds
        }
        primary_policy = f"validation_rank_ic_min_days_{primary_rank_ic_days}"
    else:
        policies = {
            "validation_accuracy": select_per_test_year(
                reports,
                kind=kind,
                min_validation_n=min_validation_n,
            )
        }
        primary_policy = "validation_accuracy"

    policy_results: dict[str, Any] = {}
    for policy_name, selection in policies.items():
        stock_day = _selected_stock_days(
            reports, selection, kind=kind, panel=panel
        )
        sensitivity: dict[str, Any] = {}
        for threshold in min_stocks:
            portfolio = form_portfolio(stock_day, min_stocks=threshold)
            sensitivity[str(threshold)] = portfolio_summary(
                portfolio,
                cost_bps=cost_bps,
                long_only_turnover=long_only_turnover,
                long_short_turnover=long_short_turnover,
                rng=np.random.default_rng(seed + threshold),
                n_bootstrap=n_bootstrap,
            )
        policy_results[policy_name] = {
            "selection": selection,
            "selected_test_years": [
                row["test_year"] for row in selection if row["status"] == "selected"
            ],
            "unselected_test_years": [
                row["test_year"] for row in selection if row["status"] != "selected"
            ],
            "stock_day_rows": int(len(stock_day)),
            "portfolio_sensitivity": sensitivity,
        }
    return {
        "candidate_reports": len(reports),
        "primary_policy": primary_policy,
        "policies": policy_results,
        "test_selection_is_validation_only": True,
        "test_metric_used_for_selection": False,
    }


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _format_percent(value: Any) -> str:
    number = _finite_float(value)
    return "NA" if not np.isfinite(number) else f"{number * 100:.2f}%"


def _format_number(value: Any, digits: int = 3) -> str:
    number = _finite_float(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable strict audit report."""
    lines = [
        "# Sina 严格无泄漏收益率审计",
        "",
        f"审计版本：`{report['selection_version']}`。所有测试年模型均只用验证期指标选择；测试收益、测试 Rank IC 和测试分类指标未参与选模。",
        "",
        "## 结论",
        "",
        "- 这些收益是无交易成本的事件日 next-day return 组合结果，不是完整持仓级回测。",
        "- 不报告机械年化收益：有效横截面日期数和日历覆盖不足，且日期并非连续交易日。",
        "- 交易成本仅作固定成本情景；当前结果没有持仓、换手、价格或完整每日市场面板，因此不能解释为可执行策略净收益。",
        "",
        "## 数据与选模",
        "",
        f"- Panel 行数：{report['data']['panel_rows']}；有效日期行：{report['data']['valid_entry_date_rows']}；有效标签行：{report['data']['valid_labeled_rows']}；测试年份：{report['data']['test_years']}。",
        f"- 最小横截面敏感性：{report['design']['min_stocks_per_day']}；日期 bootstrap 次数：{report['design']['n_bootstrap']}。",
        f"- 回归主规则：验证期 Rank IC，且有效 Rank IC 天数至少 {report['design']['primary_rank_ic_days']}；同时记录至少 1 天的敏感性结果。",
        "",
    ]
    for family, result in report["families"].items():
        lines.extend([f"## {family}", ""])
        lines.append(f"主选模规则：`{result['primary_policy']}`。")
        lines.append("")
        for policy_name, policy in result["policies"].items():
            selected = policy["selected_test_years"]
            missing = policy["unselected_test_years"]
            lines.append(
                f"### {policy_name}：选中测试年 {selected or '无'}；未选中 {missing or '无'}"
            )
            lines.append("")
            lines.append(
                "| 最小股票数 | 有效日期 | 等权基线累计 | Q5 多头累计 | Q5-Q1 多空累计 | 多空事件日均值 | 多空正收益率 | 多空符号检验 p |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for threshold, summary in policy["portfolio_sensitivity"].items():
                gross = summary["gross"]
                market = gross["market_equal_weight"]
                ls = gross["q5_minus_q1_long_short"]
                q5 = gross["q5_high_score_long_only"]
                lines.append(
                    "| {threshold} | {days} | {market_cum} | {q5_cum} | {ls_cum} | {ls_mean} | {ls_pos} | {p} |".format(
                        threshold=threshold,
                        days=summary["effective_dates"],
                        market_cum=_format_percent(market["cumulative_return"]),
                        q5_cum=_format_percent(q5["cumulative_return"]),
                        ls_cum=_format_percent(ls["cumulative_return"]),
                        ls_mean=_format_percent(ls["mean_event_day_return"]),
                        ls_pos=_format_percent(ls["positive_day_rate"]),
                        p=_format_number(ls["sign_test_p_greater_0"], 4),
                    )
                )
            lines.append("")
            lines.append(
                "| 年份 | 有效日期 | 平均股票数 | 等权基线累计 | Q5 多头累计 | Q5-Q1 多空累计 | 多空事件日均值 |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|")
            primary_threshold = str(report["design"]["min_stocks_per_day"][0])
            primary_summary = policy["portfolio_sensitivity"].get(primary_threshold)
            if primary_summary is not None:
                for year, year_summary in primary_summary["by_year"].items():
                    year_gross = year_summary["gross"]
                    lines.append(
                        "| {year} | {days} | {stocks} | {market} | {q5} | {ls} | {ls_mean} |".format(
                            year=year,
                            days=year_summary["effective_dates"],
                            stocks=_format_number(year_summary["mean_stocks_per_date"], 1),
                            market=_format_percent(
                                year_gross["market_equal_weight"]["cumulative_return"]
                            ),
                            q5=_format_percent(
                                year_gross["q5_high_score_long_only"]["cumulative_return"]
                            ),
                            ls=_format_percent(
                                year_gross["q5_minus_q1_long_short"]["cumulative_return"]
                            ),
                            ls_mean=_format_percent(
                                year_gross["q5_minus_q1_long_short"]["mean_event_day_return"]
                            ),
                        )
                    )
            lines.append("")
            lines.append(
                "成本敏感性采用固定换手假设；Q5 多头换手={long_only}，Q5-Q1 多空换手={long_short}。".format(
                    long_only=report["design"]["long_only_turnover"],
                    long_short=report["design"]["long_short_turnover"],
                )
            )
            lines.append("")
            if primary_threshold in policy["portfolio_sensitivity"]:
                costs = policy["portfolio_sensitivity"][primary_threshold]["cost_sensitivity"]
                for portfolio_name, details in costs.items():
                    scenarios = ", ".join(
                        f"{int(item['cost_bps'])}bp: {_format_percent(item['net']['cumulative_return'])}"
                        for item in details["scenarios"]
                    )
                    lines.append(f"- {portfolio_name} 净累计收益情景：{scenarios}。")
            lines.append("")
    lines.extend([
        "## 解释边界",
        "",
        "- 如果验证期有效 Rank IC 天数不足主阈值，回归结果保留为未选中，而不是用测试结果补选。当前数据因此可能没有覆盖全部测试年份。",
        "- Q5 或多空的点估计不能独立证明 alpha；应同时查看有效日期数、日期 bootstrap 区间、逐年稳定性和基线。",
        "- 当前审计支持的最强结论是：结果可作为探索性 gross event-day 证据；样本仍不足以支持可靠的策略年化收益或稳定 alpha 宣称。",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--classification-dir", type=Path, default=None)
    parser.add_argument("--gate-dir", type=Path, default=None)
    parser.add_argument("--regression-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--min-stocks-per-day", default="5,10,20")
    parser.add_argument("--cost-bps", default="0,10,25,50")
    parser.add_argument("--rank-ic-thresholds", default="1,3")
    parser.add_argument("--primary-rank-ic-days", type=int, default=3)
    parser.add_argument("--min-validation-n", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--long-only-turnover", type=float, default=1.0)
    parser.add_argument("--long-short-turnover", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.primary_rank_ic_days < 1:
        raise ValueError("primary-rank-ic-days must be positive")
    min_stocks = _parse_ints(args.min_stocks_per_day, name="min-stocks-per-day")
    cost_bps = _parse_floats(args.cost_bps, name="cost-bps")
    rank_thresholds = _parse_ints(args.rank_ic_thresholds, name="rank-ic-thresholds")
    if args.primary_rank_ic_days not in rank_thresholds:
        rank_thresholds = tuple(sorted(set(rank_thresholds + (args.primary_rank_ic_days,))))
    if args.n_bootstrap < 1:
        raise ValueError("n-bootstrap must be positive")
    if args.long_only_turnover < 0 or args.long_short_turnover < 0:
        raise ValueError("turnover assumptions cannot be negative")

    data_root = args.data_root
    panel_path = args.panel or data_root / "classification/sina_single_stock_classification_panel.parquet"
    classification_dir = args.classification_dir or data_root / "classification/results_prompt_nonqwen"
    gate_dir = args.gate_dir or data_root / "classification/results_variant_gate_nonqwen"
    regression_dir = args.regression_dir or data_root / "regression/results_prompt_nonqwen"
    output = args.output or Path(__file__).resolve().parents[1] / "reports/sina_strict_return_audit_20260812.json"
    markdown_output = args.markdown_output or output.with_suffix(".md")

    panel = load_panel(panel_path)
    classification_reports = load_reports(classification_dir)
    gate_reports = load_reports(gate_dir)
    regression_reports = load_reports(regression_dir)
    test_years = sorted({
        int(row["test_year"])
        for report in classification_reports
        for row in report.get("results", [])
        if row.get("test_year") is not None
    })
    if not test_years:
        raise ValueError("no test years found in classification reports")

    common_kwargs = {
        "panel": panel,
        "min_stocks": min_stocks,
        "cost_bps": cost_bps,
        "long_only_turnover": args.long_only_turnover,
        "long_short_turnover": args.long_short_turnover,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "rank_ic_thresholds": rank_thresholds,
        "primary_rank_ic_days": args.primary_rank_ic_days,
        "min_validation_n": args.min_validation_n,
    }
    families = {
        "classification": audit_family(
            classification_reports, kind="classification", **common_kwargs
        ),
        "gate": audit_family(gate_reports, kind="gate", **common_kwargs),
        "regression": audit_family(
            regression_reports, kind="regression", **common_kwargs
        ),
    }
    report: dict[str, Any] = {
        "selection_version": SELECTION_VERSION,
        "data": {
            "panel": str(panel_path),
            "panel_rows": int(len(panel)),
            "unique_article_ids": int(panel["article_id"].nunique()),
            "valid_entry_date_rows": int(panel["entry_date"].notna().sum()),
            "valid_labeled_rows": int(
                (panel["entry_date"].notna() & panel["next_day_return"].notna()).sum()
            ),
            "test_years": test_years,
            "classification_dir": str(classification_dir),
            "gate_dir": str(gate_dir),
            "regression_dir": str(regression_dir),
        },
        "design": {
            "selection_is_validation_only": True,
            "test_labels_used_for_selection": False,
            "min_stocks_per_day": list(min_stocks),
            "primary_rank_ic_days": args.primary_rank_ic_days,
            "rank_ic_day_sensitivity": list(rank_thresholds),
            "min_validation_n": args.min_validation_n,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_unit": "entry_date",
            "cost_bps": list(cost_bps),
            "long_only_turnover": args.long_only_turnover,
            "long_short_turnover": args.long_short_turnover,
            "annualization": "suppressed; dates are sparse event dates, not a continuous daily series",
            "return_definition": "gross equal-weighted next_day_return on selected stock-day portfolios",
            "cost_definition": "illustrative constant cost_bps times assumed turnover; no holdings-level execution data",
        },
        "families": families,
    }
    safe_report = _json_safe(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(safe_report, ensure_ascii=False, indent=2) + "\n")
    markdown_output.write_text(markdown_report(safe_report) + "\n")
    print(json.dumps({
        "output": str(output),
        "markdown_output": str(markdown_output),
        "test_years": test_years,
        "families": list(families),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()