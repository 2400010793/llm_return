"""Summarize strict CNINFO Qwen token-versus-article OOS factors."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROMPTS = ("future_return", "return", "loss", "excess_return", "profit")
KINDS = ("target_token", "article_mean")


def finite_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    values = pd.DataFrame({"left": left, "right": right}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(values) < 3 or values["left"].nunique() < 2 or values["right"].nunique() < 2:
        return float("nan")
    function = spearmanr if method == "spearman" else pearsonr
    return float(function(values["left"], values["right"]).statistic)


def daily_spearman(frame: pd.DataFrame, left: str, right: str) -> tuple[float, int]:
    values = []
    for _, group in frame.groupby("entry_date", sort=False):
        value = finite_corr(group[left], group[right], "spearman")
        if np.isfinite(value):
            values.append(value)
    return (float(np.mean(values)), len(values)) if values else (float("nan"), 0)


def portfolio_metrics(frame: pd.DataFrame, prediction: str, top_k: int) -> dict[str, float]:
    daily = []
    for date, group in frame.groupby("entry_date", sort=False):
        group = group.dropna(subset=[prediction, "actual_return"])
        if len(group) < 2 * top_k:
            continue
        ordered = group.sort_values(prediction, kind="stable")
        low = float(ordered.head(top_k)["actual_return"].mean())
        high = float(ordered.tail(top_k)["actual_return"].mean())
        daily.append((date, high, low, high - low))
    if not daily:
        return {
            "portfolio_days": 0, "top20_mean_bp": float("nan"),
            "bottom20_mean_bp": float("nan"), "long_short_mean_bp": float("nan"),
        }
    values = pd.DataFrame(daily, columns=["date", "high", "low", "long_short"])
    return {
        "portfolio_days": int(len(values)),
        "top20_mean_bp": float(values["high"].mean() * 10_000),
        "bottom20_mean_bp": float(values["low"].mean() * 10_000),
        "long_short_mean_bp": float(values["long_short"].mean() * 10_000),
    }


def top_overlap(
    frame: pd.DataFrame, left: str, right: str, top_k: int,
) -> tuple[float, float, int]:
    jaccard, overlap = [], []
    for _, group in frame.groupby("entry_date", sort=False):
        group = group.dropna(subset=[left, right])
        if len(group) < top_k:
            continue
        left_ids = set(group.nlargest(top_k, left)["stock_id"].astype(str))
        right_ids = set(group.nlargest(top_k, right)["stock_id"].astype(str))
        common = len(left_ids & right_ids)
        jaccard.append(common / max(len(left_ids | right_ids), 1))
        overlap.append(common / top_k)
    return (
        float(np.mean(jaccard)) if jaccard else float("nan"),
        float(np.mean(overlap)) if overlap else float("nan"),
        len(jaccard),
    )


def load_factor(root: Path, prompt: str, kind: str) -> tuple[pd.DataFrame, dict]:
    stem = root / f"masked_short_{prompt}_{kind}"
    report_path = stem.with_suffix(".json")
    prediction_path = stem.with_suffix(".stock_day_predictions.parquet")
    if not report_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"missing regression outputs for {prompt}/{kind}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(prediction_path)
    required = {"stock_id", "entry_date", "actual_return", "prediction"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{prediction_path} missing columns: {sorted(missing)}")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="raise")
    if frame.duplicated(["stock_id", "entry_date"]).any():
        raise ValueError(f"duplicate stock-day keys in {prediction_path}")
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    factors: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    key_sets = []
    for prompt in PROMPTS:
        for kind in KINDS:
            name = f"{prompt}_{kind}"
            factors[name], reports[name] = load_factor(args.regression_root, prompt, kind)
            key_sets.append(set(zip(factors[name]["stock_id"], factors[name]["entry_date"])))
    common_keys = set.intersection(*key_sets)
    if not common_keys:
        raise ValueError("regression outputs have no common stock-day rows")
    common = pd.DataFrame(sorted(common_keys), columns=["stock_id", "entry_date"])
    actual_reference = None
    for name, frame in factors.items():
        current = common.merge(
            frame[["stock_id", "entry_date", "actual_return", "prediction"]],
            on=["stock_id", "entry_date"], how="left", validate="one_to_one",
        )
        if actual_reference is None:
            actual_reference = current["actual_return"].to_numpy(dtype=float)
            common["actual_return"] = actual_reference
        elif not np.allclose(
            actual_reference, current["actual_return"].to_numpy(dtype=float),
            equal_nan=True, rtol=0.0, atol=0.0,
        ):
            raise ValueError(f"actual returns differ for {name}")
        common[name] = current["prediction"].to_numpy(dtype=float)
    common.to_parquet(args.output_dir / "common_oos_predictions.parquet", index=False)

    metric_rows = []
    for name in factors:
        prompt, kind = name.rsplit("_", 2)[0], None
        for candidate in KINDS:
            suffix = f"_{candidate}"
            if name.endswith(suffix):
                prompt, kind = name[:-len(suffix)], candidate
                break
        rank_ic, days = daily_spearman(common, name, "actual_return")
        result = reports[name]["results"][-1]
        metric_rows.append({
            "factor": name, "prompt": prompt, "representation": kind,
            "common_stock_days": len(common), "rank_ic_mean_recomputed": rank_ic,
            "rank_ic_days": days,
            "runner_rank_ic_mean": result.get("rank_ic_mean"),
            "runner_oos_r2": result.get("oos_r2_vs_historical_mean"),
            "validation_rank_ic": result.get("validation_metrics", {}).get("rank_ic_mean"),
            "selected_alpha": result.get("best_params", {}).get("alpha"),
            **portfolio_metrics(common, name, args.top_k),
        })
    metrics = pd.DataFrame(metric_rows).sort_values(
        "rank_ic_mean_recomputed", ascending=False
    )
    metrics.to_csv(args.output_dir / "factor_metrics.csv", index=False)

    correlation_rows = []
    overlap_rows = []
    for left, right in itertools.combinations(factors, 2):
        daily, days = daily_spearman(common, left, right)
        correlation_rows.append({
            "left": left, "right": right,
            "pooled_pearson": finite_corr(common[left], common[right], "pearson"),
            "pooled_spearman": finite_corr(common[left], common[right], "spearman"),
            "mean_daily_spearman": daily, "daily_correlation_days": days,
        })
        jaccard, overlap, overlap_days = top_overlap(common, left, right, args.top_k)
        overlap_rows.append({
            "left": left, "right": right, "top_k": args.top_k,
            "mean_jaccard": jaccard, "mean_overlap_fraction": overlap,
            "days": overlap_days,
        })
    correlations = pd.DataFrame(correlation_rows)
    overlaps = pd.DataFrame(overlap_rows)
    correlations.to_csv(args.output_dir / "factor_correlations.csv", index=False)
    overlaps.to_csv(args.output_dir / "factor_top20_overlap.csv", index=False)

    token_full_rows = []
    for prompt in PROMPTS:
        left, right = f"{prompt}_target_token", f"{prompt}_article_mean"
        correlation = correlations[
            ((correlations["left"] == left) & (correlations["right"] == right))
            | ((correlations["left"] == right) & (correlations["right"] == left))
        ].iloc[0]
        overlap = overlaps[
            ((overlaps["left"] == left) & (overlaps["right"] == right))
            | ((overlaps["left"] == right) & (overlaps["right"] == left))
        ].iloc[0]
        left_metric = metrics.set_index("factor").loc[left]
        right_metric = metrics.set_index("factor").loc[right]
        token_full_rows.append({
            "prompt": prompt,
            "token_rank_ic": left_metric["rank_ic_mean_recomputed"],
            "article_rank_ic": right_metric["rank_ic_mean_recomputed"],
            "token_minus_article_rank_ic": (
                left_metric["rank_ic_mean_recomputed"] - right_metric["rank_ic_mean_recomputed"]
            ),
            "prediction_pooled_spearman": correlation["pooled_spearman"],
            "prediction_mean_daily_spearman": correlation["mean_daily_spearman"],
            "top20_overlap_fraction": overlap["mean_overlap_fraction"],
            "token_top20_bp": left_metric["top20_mean_bp"],
            "article_top20_bp": right_metric["top20_mean_bp"],
        })
    token_full = pd.DataFrame(token_full_rows)
    token_full.to_csv(args.output_dir / "token_vs_article.csv", index=False)

    raw_path = args.geometry_root / "raw_embedding_pairwise.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    lines = [
        "# Qwen 巨潮 Token 与全文严格比较", "",
        "## 设计", "",
        f"- 严格共同样本；masked_short；PCA-128 + Ridge；共同测试股票日 {len(common):,} 条。",
        "- 6 年训练、2 年验证、1 年测试；当前共同覆盖只形成 2026 年一个测试折。",
        "- 所有预测相关性和 Top20 重合均在十个模型的完全共同股票日上重新计算。", "",
        "## Token 对全文", "",
        token_full.to_markdown(index=False, floatfmt=".6f"), "",
        "## 样本外指标", "",
        metrics.to_markdown(index=False, floatfmt=".6f"), "",
    ]
    if not raw.empty:
        lines.extend(["## 原始表示几何", "", raw.to_markdown(index=False, floatfmt=".6f"), ""])
    lines.extend([
        "## 输出", "",
        "- `factor_correlations.csv`: 十个样本外预测因子的两两相关。",
        "- `factor_top20_overlap.csv`: 每日 Top20 选股重合。",
        "- `common_oos_predictions.parquet`: 完全共同测试面板。", "",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "format_version": "qwen_cninfo_strict_comparison_v1",
        "prompts": list(PROMPTS), "representations": list(KINDS),
        "common_stock_days": len(common), "top_k": args.top_k,
        "test_years": sorted(pd.to_datetime(common["entry_date"]).dt.year.unique().tolist()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(token_full.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
