"""Compare single-variant and variant-gate classifiers without test-set selection.

The existing Sina gate runs use Logistic regression, while the first prompt
comparison used HistGradientBoosting.  This audit consumes a matching Logistic
baseline for each of the five variants and compares, within each fixed
embedding model:

* the validation-selected single variant;
* the fixed uniform gate;
* the validation-selected learned gate among variance, Fisher, L1 logistic,
  and dynamic gates.

All selections use validation accuracy only.  Test predictions are used only
after the selection records have been frozen.  Article-level paired accuracy
comparisons use exact McNemar tests and date-cluster bootstrap intervals;
portfolio comparisons use the same event dates and date-level bootstrap.
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

from scripts.audit_sina_strict_returns import (
    _sign_test_pvalue,
    aggregate_stock_day,
    form_portfolio,
    load_candidate_predictions,
    load_panel,
    load_reports,
    portfolio_summary,
    select_per_test_year,
)


EMBEDDING_MODELS = ("chinese_bert", "chinese_roberta", "bge_m3", "word2vec")
VARIANTS = ("plain", "prompt_short", "prompt_long", "masked_short", "masked_long")
LEARNED_GATE_MODES = ("variance", "fisher", "logistic_l1", "dynamic")
DEFAULT_MIN_STOCKS = (5, 10, 20)
DEFAULT_N_BOOTSTRAP = 5000
AUDIT_VERSION = "sina_fair_gate_comparison_v1"


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _model_name(report: Mapping[str, Any]) -> str:
    experiment = report.get("experiment", {})
    value = experiment.get("embedding_model")
    if value:
        return str(value)
    return str(report.get("_name", "")).split("_", 1)[0]


def _variant_name(report: Mapping[str, Any]) -> str | None:
    experiment = report.get("experiment", {})
    input_name = experiment.get("input_name")
    if isinstance(input_name, str) and input_name.startswith("sina_"):
        return input_name.removeprefix("sina_")
    name = str(report.get("_name", ""))
    for variant in VARIANTS:
        if f"_{variant}_" in f"_{name}_":
            return variant
    return None


def _gate_mode(report: Mapping[str, Any]) -> str | None:
    value = report.get("experiment", {}).get("gate_mode")
    if value:
        return str(value)
    name = str(report.get("_name", ""))
    for mode in ("logistic_l1", "uniform", "variance", "fisher", "dynamic"):
        if f"_{mode}_" in f"_{name}_":
            return mode
    return None


def _reports_for_model(
    reports: Iterable[Mapping[str, Any]], model: str
) -> list[dict[str, Any]]:
    return [dict(report) for report in reports if _model_name(report) == model]


def _validate_candidate_set(
    reports: list[Mapping[str, Any]],
    *,
    model: str,
    kind: str,
    expected: Iterable[str],
) -> None:
    if not reports:
        raise ValueError(f"no {kind} reports found for embedding model {model}")
    found = {
        _variant_name(report) if kind == "single_variant" else _gate_mode(report)
        for report in reports
    }
    expected_set = set(expected)
    missing = sorted(expected_set - found)
    if missing:
        raise ValueError(
            f"{model} {kind} reports are incomplete; missing {', '.join(missing)}"
        )


def fixed_selection(
    reports: Iterable[Mapping[str, Any]],
    *,
    selector: str,
    value: str,
) -> list[dict[str, Any]]:
    """Create a fixed, non-test-selected selection for one candidate."""
    candidates = [
        report for report in reports
        if (selector == "variant" and _variant_name(report) == value)
        or (selector == "gate" and _gate_mode(report) == value)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one fixed {selector} candidate {value}, found {len(candidates)}"
        )
    report = candidates[0]
    years = sorted(
        int(row["test_year"])
        for row in report.get("results", [])
        if row.get("test_year") is not None
    )
    return [
        {
            "test_year": year,
            "status": "selected",
            "selected_model": report["_name"],
            "selected_report": report["_path"],
            "eligible_count": 1,
            "candidate_count": 1,
            "selection_basis": f"fixed_{selector}_{value}",
        }
        for year in years
    ]


def select_method_candidates(
    reports: list[Mapping[str, Any]],
    *,
    kind: str,
    min_validation_n: int,
) -> list[dict[str, Any]]:
    """Select only from a pre-declared within-method candidate set."""
    return select_per_test_year(
        reports,
        kind="classification",
        min_validation_n=min_validation_n,
    )


def _selected_report_map(
    reports: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {str(report["_name"]): report for report in reports}


def selected_predictions(
    reports: Iterable[Mapping[str, Any]],
    selection: Iterable[Mapping[str, Any]],
    *,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Load the selected test predictions, retaining article-level rows."""
    by_name = _selected_report_map(reports)
    frames: list[pd.DataFrame] = []
    for row in selection:
        if row.get("status") != "selected":
            continue
        model_name = str(row["selected_model"])
        report = by_name[model_name]
        values = load_candidate_predictions(
            report,
            kind="classification",
            test_year=int(row["test_year"]),
            panel=panel,
        ).copy()
        values["selected_model"] = model_name
        frames.append(values)
    if not frames:
        return pd.DataFrame(
            columns=[
                "article_id", "stock_id", "entry_date", "actual_return",
                "score", "test_year", "selected_model",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _date_cluster_bootstrap_mean(
    values: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "entry_date",
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[float | None]:
    """Bootstrap a mean by resampling whole entry dates."""
    frame = values[[date_column, value_column]].dropna().copy()
    if frame.empty or n_bootstrap < 1:
        return [None, None]
    clusters = [
        group[value_column].to_numpy(dtype=float)
        for _, group in frame.groupby(date_column)
    ]
    if not clusters:
        return [None, None]
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        draws[index] = np.concatenate([clusters[item] for item in selected]).mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _exact_mcnemar_pvalue(discordant_a: int, discordant_b: int) -> float:
    total = int(discordant_a + discordant_b)
    if total == 0:
        return 1.0
    lower = min(discordant_a, discordant_b)
    probability = sum(math.comb(total, index) for index in range(lower + 1)) / 2**total
    return float(min(1.0, 2.0 * probability))


def paired_classification_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Compare two selected classifiers on exactly matched articles."""
    required = {"article_id", "entry_date", "actual_return", "score"}
    if not required.issubset(left.columns) or not required.issubset(right.columns):
        raise ValueError("paired classification inputs lack required columns")
    left_values = left[["article_id", "entry_date", "actual_return", "score"]].rename(
        columns={"score": "score_left"}
    )
    right_values = right[["article_id", "entry_date", "actual_return", "score"]].rename(
        columns={
            "entry_date": "entry_date_right",
            "score": "score_right",
            "actual_return": "actual_return_right",
        }
    )
    merged = left_values.merge(
        right_values,
        on="article_id",
        how="inner",
        validate="one_to_one",
    )
    merged["entry_date"] = pd.to_datetime(merged["entry_date"], errors="coerce")
    merged["entry_date_right"] = pd.to_datetime(
        merged["entry_date_right"], errors="coerce"
    )
    merged["actual_return"] = pd.to_numeric(merged["actual_return"], errors="coerce")
    merged["actual_return_right"] = pd.to_numeric(
        merged["actual_return_right"], errors="coerce"
    )
    merged["score_left"] = pd.to_numeric(merged["score_left"], errors="coerce")
    merged["score_right"] = pd.to_numeric(merged["score_right"], errors="coerce")
    merged = merged.dropna(
        subset=[
            "entry_date", "entry_date_right", "actual_return",
            "actual_return_right", "score_left", "score_right",
        ]
    )
    if not np.array_equal(
        merged["entry_date"].to_numpy(), merged["entry_date_right"].to_numpy()
    ):
        raise ValueError("paired classifiers have inconsistent entry_date values")
    if not np.allclose(
        merged["actual_return"].to_numpy(dtype=float),
        merged["actual_return_right"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("paired classifiers have inconsistent actual returns")
    actual = merged["actual_return"].to_numpy(dtype=float) > 0
    left_correct = (merged["score_left"].to_numpy(dtype=float) >= 0.5) == actual
    right_correct = (merged["score_right"].to_numpy(dtype=float) >= 0.5) == actual
    merged["correct_difference_right_minus_left"] = (
        right_correct.astype(np.int8) - left_correct.astype(np.int8)
    )
    discordant_left_only = int(np.sum(left_correct & ~right_correct))
    discordant_right_only = int(np.sum(~left_correct & right_correct))
    left_accuracy = float(left_correct.mean()) if len(left_correct) else None
    right_accuracy = float(right_correct.mean()) if len(right_correct) else None
    difference = (
        float(right_accuracy - left_accuracy)
        if left_accuracy is not None and right_accuracy is not None
        else None
    )
    return {
        "n_articles": int(len(merged)),
        "n_dates": int(merged["entry_date"].nunique()),
        "left_accuracy": left_accuracy,
        "right_accuracy": right_accuracy,
        "accuracy_difference_right_minus_left": difference,
        "date_cluster_bootstrap_95_ci_difference": _date_cluster_bootstrap_mean(
            merged,
            value_column="correct_difference_right_minus_left",
            rng=rng,
            n_bootstrap=n_bootstrap,
        ),
        "mcnemar": {
            "left_correct_right_wrong": discordant_left_only,
            "left_wrong_right_correct": discordant_right_only,
            "exact_two_sided_p": _exact_mcnemar_pvalue(
                discordant_left_only, discordant_right_only
            ),
        },
    }


def _paired_return_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    return_column: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    merged = left[["date", return_column]].merge(
        right[["date", return_column]],
        on="date",
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    merged["difference_right_minus_left"] = (
        merged[f"{return_column}_right"] - merged[f"{return_column}_left"]
    )
    values = pd.to_numeric(merged["difference_right_minus_left"], errors="coerce")
    merged = merged.loc[values.notna()].copy()
    values = merged["difference_right_minus_left"].to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "n_common_dates": 0,
            "mean_difference_right_minus_left": None,
            "positive_difference_rate": None,
            "sign_test_p_greater_0": None,
            "date_bootstrap_95_ci_mean_difference": [None, None],
        }
    return {
        "n_common_dates": int(len(values)),
        "mean_difference_right_minus_left": float(values.mean()),
        "positive_difference_rate": float(np.mean(values > 0)),
        "sign_test_p_greater_0": _sign_test_pvalue(values),
        "date_bootstrap_95_ci_mean_difference": _date_cluster_bootstrap_mean(
            merged.rename(columns={"difference_right_minus_left": "value"}),
            value_column="value",
            date_column="date",
            rng=rng,
            n_bootstrap=n_bootstrap,
        ),
    }


def _portfolio_for_predictions(
    predictions: pd.DataFrame, min_stocks: Iterable[int]
) -> dict[str, pd.DataFrame]:
    stock_day = aggregate_stock_day(predictions)
    return {
        str(threshold): form_portfolio(stock_day, min_stocks=threshold)
        for threshold in min_stocks
    }


def paired_portfolio_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    min_stocks: tuple[int, ...],
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Compare portfolios on identical dates for one pooled or yearly slice."""
    left_portfolios = _portfolio_for_predictions(left, min_stocks)
    right_portfolios = _portfolio_for_predictions(right, min_stocks)
    portfolios: dict[str, Any] = {}
    for threshold_index, threshold in enumerate(min_stocks):
        portfolios[str(threshold)] = {
            "q5_long_only": _paired_return_summary(
                left_portfolios[str(threshold)],
                right_portfolios[str(threshold)],
                return_column="q5",
                rng=np.random.default_rng(seed + threshold_index * 10 + 2),
                n_bootstrap=n_bootstrap,
            ),
            "q5_minus_q1_long_short": _paired_return_summary(
                left_portfolios[str(threshold)],
                right_portfolios[str(threshold)],
                return_column="long_short",
                rng=np.random.default_rng(seed + threshold_index * 10 + 3),
                n_bootstrap=n_bootstrap,
            ),
        }
    return portfolios


def _method_selections(
    baseline: list[Mapping[str, Any]],
    gate: list[Mapping[str, Any]],
    *,
    model: str,
    min_validation_n: int,
) -> dict[str, list[dict[str, Any]]]:
    model_baseline = _reports_for_model(baseline, model)
    model_gate = _reports_for_model(gate, model)
    _validate_candidate_set(
        model_baseline,
        model=model,
        kind="single_variant",
        expected=VARIANTS,
    )
    _validate_candidate_set(
        model_gate,
        model=model,
        kind="gate",
        expected=("uniform", *LEARNED_GATE_MODES),
    )
    single_variant = [
        report for report in model_baseline if _variant_name(report) in VARIANTS
    ]
    uniform = [
        report for report in model_gate if _gate_mode(report) == "uniform"
    ]
    learned = [
        report for report in model_gate if _gate_mode(report) in LEARNED_GATE_MODES
    ]
    return {
        "single_variant": select_method_candidates(
            single_variant,
            kind="single_variant",
            min_validation_n=min_validation_n,
        ),
        "uniform_gate": fixed_selection(
            uniform,
            selector="gate",
            value="uniform",
        ),
        "learned_gate": select_method_candidates(
            learned,
            kind="learned_gate",
            min_validation_n=min_validation_n,
        ),
    }


def audit_model(
    baseline: list[Mapping[str, Any]],
    gate: list[Mapping[str, Any]],
    *,
    model: str,
    panel: pd.DataFrame,
    min_stocks: tuple[int, ...],
    min_validation_n: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    selections = _method_selections(
        baseline,
        gate,
        model=model,
        min_validation_n=min_validation_n,
    )
    model_baseline = _reports_for_model(baseline, model)
    model_gate = _reports_for_model(gate, model)
    predictions = {
        "single_variant": selected_predictions(
            model_baseline, selections["single_variant"], panel=panel
        ),
        "uniform_gate": selected_predictions(
            model_gate, selections["uniform_gate"], panel=panel
        ),
        "learned_gate": selected_predictions(
            model_gate, selections["learned_gate"], panel=panel
        ),
    }
    absolute_portfolios: dict[str, Any] = {}
    for method_index, (method, frame) in enumerate(predictions.items()):
        stock_day = aggregate_stock_day(frame)
        absolute_portfolios[method] = {}
        for threshold_index, threshold in enumerate(min_stocks):
            portfolio = form_portfolio(stock_day, min_stocks=threshold)
            absolute_portfolios[method][str(threshold)] = portfolio_summary(
                portfolio,
                cost_bps=(0.0,),
                long_only_turnover=1.0,
                long_short_turnover=2.0,
                rng=np.random.default_rng(
                    seed + method_index * 1000 + threshold_index * 10 + 50
                ),
                n_bootstrap=n_bootstrap,
            )
    comparison_pairs = (
        ("single_variant", "uniform_gate"),
        ("single_variant", "learned_gate"),
        ("uniform_gate", "learned_gate"),
    )
    comparisons: dict[str, Any] = {}
    for pair_index, (left_name, right_name) in enumerate(comparison_pairs):
        left = predictions[left_name]
        right = predictions[right_name]
        article = paired_classification_comparison(
            left,
            right,
            rng=np.random.default_rng(seed + pair_index * 1000 + 1),
            n_bootstrap=n_bootstrap,
        )
        portfolios = paired_portfolio_comparison(
            left,
            right,
            min_stocks=min_stocks,
            seed=seed + pair_index * 1000,
            n_bootstrap=n_bootstrap,
        )
        by_test_year: dict[str, Any] = {}
        years = sorted(
            set(pd.to_numeric(left.get("test_year"), errors="coerce").dropna().astype(int))
            | set(pd.to_numeric(right.get("test_year"), errors="coerce").dropna().astype(int))
        )
        for year in years:
            left_year = left[left["test_year"] == year]
            right_year = right[right["test_year"] == year]
            by_test_year[str(year)] = {
                "article_level": paired_classification_comparison(
                    left_year,
                    right_year,
                    rng=np.random.default_rng(
                        seed + pair_index * 1000 + year * 10 + 4
                    ),
                    n_bootstrap=n_bootstrap,
                ),
                "portfolio_level": paired_portfolio_comparison(
                    left_year,
                    right_year,
                    min_stocks=min_stocks,
                    seed=seed + pair_index * 1000 + year * 10 + 5,
                    n_bootstrap=n_bootstrap,
                ),
            }
        comparisons[f"{left_name}_vs_{right_name}"] = {
            "left_method": left_name,
            "right_method": right_name,
            "difference_definition": "right minus left",
            "article_level": article,
            "portfolio_level": portfolios,
            "by_test_year": by_test_year,
        }
    return {
        "selection": selections,
        "selected_test_years": {
            method: [
                row["test_year"]
                for row in selection
                if row.get("status") == "selected"
            ]
            for method, selection in selections.items()
        },
        "predictions_rows": {
            method: int(len(frame)) for method, frame in predictions.items()
        },
        "absolute_portfolio_summary_gross": absolute_portfolios,
        "comparisons": comparisons,
    }


def _json_safe(value: Any) -> Any:
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


def _percent(value: Any) -> str:
    number = _finite_float(value)
    return "NA" if not np.isfinite(number) else f"{number * 100:.3f}%"


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Sina 公平 Logistic 门控对照审计",
        "",
        f"审计版本：`{report['audit_version']}`。每个 embedding 内部的候选模型只按验证期 Accuracy 选择；测试标签只用于最终配对比较。",
        "",
        "## 设计边界",
        "",
        "- `single_variant`：五种文本变体中按验证 Accuracy 选出的单一变体。",
        "- `uniform_gate`：固定的五变体等权门控，不在测试集选模。",
        "- `learned_gate`：在 variance、Fisher、L1 logistic、dynamic 四种门控中按验证 Accuracy 选模。",
        "- 每个 embedding 单独报告，避免用测试收益选择 embedding；所有方法均使用 Logistic、无 PCA、相同测试日期和股票面板。",
        "- 分类器比较使用文章级 McNemar；组合比较使用相同事件日期的日期级 bootstrap。",
        "",
    ]
    for model, result in report["models"].items():
        lines.extend([f"## {model}", ""])
        lines.append("### 逐年验证选模")
        lines.append("")
        lines.append("| 方法 | 测试年 | 选中候选 |")
        lines.append("|---|---:|---|")
        for method, selection in result["selection"].items():
            for row in selection:
                lines.append(
                    f"| {method} | {row['test_year']} | {row.get('selected_model') or '未选中'} |"
                )
        lines.extend([
            "",
            "### 绝对组合结果（gross）",
            "",
            "| 方法 | 股票数阈值 | 有效日期 | Q5 累计 | Q5-Q1 多空累计 |",
            "|---|---:|---:|---:|---:|",
        ])
        for method, summaries in result["absolute_portfolio_summary_gross"].items():
            for threshold, summary in summaries.items():
                gross = summary["gross"]
                lines.append(
                    f"| {method} | {threshold} | {summary['effective_dates']} | {_percent(gross['q5_high_score_long_only']['cumulative_return'])} | {_percent(gross['q5_minus_q1_long_short']['cumulative_return'])} |"
                )
        lines.extend(["", "### 相对收益与分类差异", ""])
        lines.append(
            "以下均为右侧方法减左侧方法；收益差异是相同事件日期上的平均 next-day return 差异，不是累计策略收益。"
        )
        lines.append("")
        lines.append(
            "| 比较 | 股票数阈值 | Q5 平均差异 | Q5 差异 95% CI | 多空平均差异 | 多空差异 95% CI | 共同日期 |"
        )
        lines.append("|---|---:|---:|---|---:|---|---:|")
        for name, comparison in result["comparisons"].items():
            for threshold, portfolio in comparison["portfolio_level"].items():
                q5 = portfolio["q5_long_only"]
                long_short = portfolio["q5_minus_q1_long_short"]
                q5_ci = q5["date_bootstrap_95_ci_mean_difference"]
                ls_ci = long_short["date_bootstrap_95_ci_mean_difference"]
                lines.append(
                    "| {name} | {threshold} | {q5_mean} | [{q5_lo}, {q5_hi}] | {ls_mean} | [{ls_lo}, {ls_hi}] | {n} |".format(
                        name=name,
                        threshold=threshold,
                        q5_mean=_percent(q5["mean_difference_right_minus_left"]),
                        q5_lo=_percent(q5_ci[0]),
                        q5_hi=_percent(q5_ci[1]),
                        ls_mean=_percent(long_short["mean_difference_right_minus_left"]),
                        ls_lo=_percent(ls_ci[0]),
                        ls_hi=_percent(ls_ci[1]),
                        n=long_short["n_common_dates"],
                    )
                )
        lines.extend([
            "",
            "| 比较 | 共同文章 | Accuracy 差异（右−左） | 日期 bootstrap 95% CI | McNemar p |",
            "|---|---:|---:|---|---:|",
        ])
        for name, comparison in result["comparisons"].items():
            article = comparison["article_level"]
            lines.append(
                f"| {name} | {article['n_articles']} | {_percent(article['accuracy_difference_right_minus_left'])} | [{_percent(article['date_cluster_bootstrap_95_ci_difference'][0])}, {_percent(article['date_cluster_bootstrap_95_ci_difference'][1])}] | {_percent(article['mcnemar']['exact_two_sided_p'])} |"
            )
        lines.extend([
            "",
            "### 按测试年（组合阈值取最小设定）",
            "",
            "| 比较 | 测试年 | 共同文章 | Accuracy 差异（右−左） | McNemar p | 多空平均收益差异 | 共同日期 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        primary_threshold = str(report["design"]["min_stocks_per_day"][0])
        for name, comparison in result["comparisons"].items():
            for year, yearly in comparison["by_test_year"].items():
                article = yearly["article_level"]
                long_short = yearly["portfolio_level"][primary_threshold][
                    "q5_minus_q1_long_short"
                ]
                lines.append(
                    f"| {name} | {year} | {article['n_articles']} | {_percent(article['accuracy_difference_right_minus_left'])} | {_percent(article['mcnemar']['exact_two_sided_p'])} | {_percent(long_short['mean_difference_right_minus_left'])} | {long_short['n_common_dates']} |"
                )
        lines.append("")
    lines.extend([
        "## 解释边界",
        "",
        "- 该审计解决的是同一 Logistic 分类器下的公平门控比较，不解决有效日期过少、样本外年份覆盖不足或持仓级执行问题。",
        "- 正的配对差异只表示门控相对基线的事件日平均改善；仍需先扩大连续横截面，再进行持仓级 T+1 回测。",
        "- 不应根据本报告的测试收益重新选择 embedding、门控方式或文本变体。",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--min-stocks-per-day", default="5,10,20")
    parser.add_argument("--min-validation-n", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _parse_ints(value: str, name: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or any(item < 1 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.min_validation_n < 1 or args.n_bootstrap < 1:
        raise ValueError("validation sample and bootstrap counts must be positive")
    min_stocks = _parse_ints(args.min_stocks_per_day, "min-stocks-per-day")
    panel = load_panel(args.panel)
    baseline = load_reports(args.baseline_dir, expected_count=20)
    gate = load_reports(args.gate_dir, expected_count=20)
    models = {
        model
        for report in baseline + gate
        if (model := _model_name(report)) in EMBEDDING_MODELS
    }
    missing_models = sorted(set(EMBEDDING_MODELS) - models)
    if missing_models:
        raise ValueError(f"missing embedding model reports: {', '.join(missing_models)}")
    results = {
        model: audit_model(
            baseline,
            gate,
            model=model,
            panel=panel,
            min_stocks=min_stocks,
            min_validation_n=args.min_validation_n,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + index * 10000,
        )
        for index, model in enumerate(EMBEDDING_MODELS)
    }
    report = _json_safe({
        "audit_version": AUDIT_VERSION,
        "design": {
            "selection_is_validation_only": True,
            "test_labels_used_for_selection": False,
            "embedding_models_fixed_and_reported_separately": list(EMBEDDING_MODELS),
            "classifier": "logistic",
            "reducer": "none",
            "min_stocks_per_day": list(min_stocks),
            "min_validation_n": args.min_validation_n,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_unit": "entry_date",
            "difference_definition": "right_method_minus_left_method",
        },
        "data": {
            "panel": str(args.panel),
            "baseline_dir": str(args.baseline_dir),
            "gate_dir": str(args.gate_dir),
        },
        "models": results,
    })
    markdown_output = args.markdown_output or args.output.with_suffix(".md")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(markdown_report(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "markdown_output": str(markdown_output),
        "models": list(results),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()