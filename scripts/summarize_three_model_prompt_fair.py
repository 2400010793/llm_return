#!/usr/bin/env python3
"""Summarize strict three-model/four-prompt folds and update the master report."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MODELS = ("roberta", "bge_m3", "qwen3_embedding_8b")
PROMPTS = ("profit", "return", "excess_return", "loss")


def daily_ic(frame: pd.DataFrame, score: str = "prediction") -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("entry_date", sort=True):
        group = group.dropna(subset=[score, "actual_return"])
        if len(group) >= 5 and group[score].nunique() > 1:
            value = spearmanr(group[score], group.actual_return).statistic
            if np.isfinite(value): rows.append({"entry_date": date, "rank_ic": float(value)})
    return pd.DataFrame(rows)


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    daily = daily_ic(frame)
    monthly = daily.assign(month=pd.to_datetime(daily.entry_date).dt.to_period("M")).groupby("month").rank_ic.mean()
    spreads, longs = [], []
    for _, group in frame.groupby("entry_date"):
        if len(group) < 5: continue
        count = max(1, int(np.ceil(len(group) * 0.2)))
        ordered = group.sort_values(["prediction", "stock_id"], kind="mergesort")
        high, low = ordered.tail(count).actual_return.mean(), ordered.head(count).actual_return.mean()
        longs.append(float(high)); spreads.append(float(high - low))
    values = daily.rank_ic.to_numpy(float)
    return {
        "rank_ic": float(values.mean()) if len(values) else np.nan,
        "rank_ic_ir": float(values.mean() / values.std(ddof=1)) if len(values) > 1 and values.std(ddof=1) else np.nan,
        "rank_ic_days": len(values), "positive_months": int((monthly > 0).sum()), "months": len(monthly),
        "top20_return": float(np.mean(longs)) if longs else np.nan,
        "top20_long_short": float(np.mean(spreads)) if spreads else np.nan,
    }


def portfolio(frame: pd.DataFrame, cost: float = 0.0005) -> dict[str, float]:
    previous_long, previous_ls = {}, {}
    long_returns, ls_returns = [], []
    for date, group in frame.groupby("entry_date", sort=True):
        if len(group) < 5: continue
        count = max(1, int(np.ceil(len(group) * 0.2)))
        ordered = group.sort_values(["prediction", "stock_id"], kind="mergesort")
        low, high = ordered.head(count), ordered.tail(count)
        long_weights = dict.fromkeys(high.stock_id.astype(str), 1.0 / count)
        ls_weights = {**dict.fromkeys(high.stock_id.astype(str), 1.0 / count),
                      **dict.fromkeys(low.stock_id.astype(str), -1.0 / count)}
        long_turnover = sum(abs(long_weights.get(key, 0) - previous_long.get(key, 0)) for key in set(long_weights) | set(previous_long))
        ls_turnover = 0.5 * sum(abs(ls_weights.get(key, 0) - previous_ls.get(key, 0)) for key in set(ls_weights) | set(previous_ls))
        long_returns.append(float(high.actual_return.mean()) - cost * long_turnover)
        ls_returns.append(float(high.actual_return.mean() - low.actual_return.mean()) - cost * ls_turnover)
        previous_long, previous_ls = long_weights, ls_weights
    def stats(values):
        values = np.asarray(values, dtype=float)
        wealth = np.cumprod(1 + values)
        peak = np.maximum.accumulate(wealth)
        years = len(values) / 252
        return {
            "annualized_return": float(wealth[-1] ** (1 / years) - 1) if len(values) and years else np.nan,
            "sharpe": float(values.mean() / values.std(ddof=1) * np.sqrt(252)) if len(values) > 1 and values.std(ddof=1) else np.nan,
            "max_drawdown": float(np.min(wealth / peak - 1)) if len(values) else np.nan,
        }
    return {f"long_{key}": value for key, value in stats(long_returns).items()} | {
        f"long_short_{key}": value for key, value in stats(ls_returns).items()
    }


def paired_block_ci(left: pd.DataFrame, right: pd.DataFrame, reps: int = 2000):
    a, b = daily_ic(left), daily_ic(right)
    values = a.merge(b, on="entry_date", suffixes=("_left", "_right"))
    delta = (values.rank_ic_left - values.rank_ic_right).to_numpy(float)
    if len(delta) < 2: return np.nan, np.nan, np.nan
    rng, block = np.random.default_rng(42), min(20, len(delta))
    starts = np.arange(max(1, len(delta) - block + 1)); blocks = int(np.ceil(len(delta) / block))
    boot = np.empty(reps)
    for rep in range(reps):
        sample = np.concatenate([delta[start:start + block] for start in rng.choice(starts, blocks)])[:len(delta)]
        boot[rep] = sample.mean()
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)


def load_base(root: Path, test_years: list[int]):
    frames, metric_frames = {}, []
    for model in MODELS:
        for prompt in PROMPTS:
            for representation in ("token", "body"):
                key = f"{model}__{prompt}__{representation}"
                pieces = []
                for year in test_years:
                    fold = root / "rolling" / model / prompt / representation / str(year)
                    if not (fold / "COMPLETED").is_file(): raise FileNotFoundError(fold / "COMPLETED")
                    metrics = pd.read_csv(fold / "metrics.csv"); metric_frames.append(metrics)
                    predictions = pd.read_parquet(fold / "test_stock_day_predictions.parquet")
                    predictions = predictions[predictions.method.eq("ridge_pca32")].copy()
                    predictions["test_year"] = year; pieces.append(predictions)
                frames[key] = pd.concat(pieces, ignore_index=True)
    return frames, pd.concat(metric_frames, ignore_index=True)


def factor_agreement(frames: dict[str, pd.DataFrame]):
    rows = []
    for model in MODELS:
        keys = [f"{model}__{prompt}__token" for prompt in PROMPTS]
        for left, right in combinations(keys, 2):
            merged = frames[left].merge(frames[right], on=["stock_id", "entry_date"], suffixes=("_left", "_right"))
            overlaps = []
            for _, group in merged.groupby("entry_date"):
                count = max(1, int(np.ceil(len(group) * 0.2)))
                a = set(group.nlargest(count, "prediction_left").stock_id)
                b = set(group.nlargest(count, "prediction_right").stock_id)
                overlaps.append(len(a & b) / count)
            rows.append({"model": model, "left": left.split("__")[1], "right": right.split("__")[1],
                         "spearman": merged[["prediction_left", "prediction_right"]].corr(method="spearman").iloc[0, 1],
                         "top20_overlap": float(np.mean(overlaps))})
    return pd.DataFrame(rows)


def all_factor_comparisons(frames: dict[str, pd.DataFrame]):
    keys = sorted(frames)
    merged = None
    for key in keys:
        item = frames[key][["stock_id", "entry_date", "prediction"]].rename(columns={"prediction": key})
        merged = item if merged is None else merged.merge(item, on=["stock_id", "entry_date"], how="inner", validate="one_to_one")
    correlation = merged[keys].corr(method="spearman")
    overlaps = []
    for left, right in combinations(keys, 2):
        daily = []
        for _, group in merged.groupby("entry_date"):
            count = max(1, int(np.ceil(len(group) * 0.2)))
            a = set(group.nlargest(count, left).stock_id)
            b = set(group.nlargest(count, right).stock_id)
            daily.append(len(a & b) / count)
        overlaps.append({"left": left, "right": right, "top20_overlap": float(np.mean(daily))})
    return correlation, pd.DataFrame(overlaps)


def load_fusion(root: Path, test_years: list[int]):
    pieces = []
    for year in test_years:
        fold = root / "fusion" / str(year)
        if not (fold / "COMPLETED").is_file(): raise FileNotFoundError(fold / "COMPLETED")
        frame = pd.read_parquet(fold / "test_stock_day_predictions.parquet")
        frame["test_year"] = year; pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def update_report(path: Path, section: str, dataset: str):
    marker = dataset.upper()
    start = f"<!-- THREE_MODEL_FOUR_PROMPT_FAIR_PCA_V2_{marker}_START -->"
    end = f"<!-- THREE_MODEL_FOUR_PROMPT_FAIR_PCA_V2_{marker}_END -->"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{start}\n{section.rstrip()}\n{end}"
    if start in current and end in current:
        prefix, remainder = current.split(start, 1)
        _, suffix = remainder.split(end, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = current.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "summary"; output.mkdir(parents=True, exist_ok=True)
    root_manifest = json.loads(
        (args.root / "intersection" / "manifest.json").read_text(encoding="utf-8")
    )
    dataset = root_manifest["dataset"]
    test_years = [int(year) for year in root_manifest["test_years"]]
    frames, all_fold_metrics = load_base(args.root, test_years)
    all_fold_metrics.to_csv(output / "all_fold_metrics.csv", index=False)
    base_summary = all_fold_metrics.groupby(["model", "prompt", "representation", "method"], as_index=False).agg(
        rank_ic=("rank_ic", "mean"), rank_ic_ir=("rank_ic_ir", "mean"),
        positive_years=("rank_ic", lambda x: int((x > 0).sum())), years=("test_year", "nunique"),
        top20_return=("top20_return", "mean"), top20_long_short=("top20_long_short", "mean"),
    )
    base_summary.to_csv(output / "base_factor_summary.csv", index=False)
    agreement = factor_agreement(frames); agreement.to_csv(output / "within_model_prompt_agreement.csv", index=False)
    factor_correlation, factor_overlap = all_factor_comparisons(frames)
    factor_correlation.to_csv(output / "all_24_factor_spearman.csv")
    factor_overlap.to_csv(output / "all_24_factor_top20_overlap.csv", index=False)
    base_detailed, base_portfolios = [], []
    for key, frame in frames.items():
        model, prompt, representation = key.split("__")
        metrics = prediction_metrics(frame)
        base_detailed.append({"model": model, "prompt": prompt, "representation": representation, **metrics})
        base_portfolios.append({"model": model, "prompt": prompt, "representation": representation, **portfolio(frame)})
    pd.DataFrame(base_detailed).to_csv(output / "base_factor_oos_metrics.csv", index=False)
    pd.DataFrame(base_portfolios).to_csv(output / "base_factor_portfolio_5bp.csv", index=False)
    token_body = []
    for model in MODELS:
        for prompt in PROMPTS:
            token = all_fold_metrics[(all_fold_metrics.model == model) & (all_fold_metrics.prompt == prompt) &
                                     (all_fold_metrics.representation == "token") & (all_fold_metrics.method == "ridge_pca32")]
            body = all_fold_metrics[(all_fold_metrics.model == model) & (all_fold_metrics.prompt == prompt) &
                                    (all_fold_metrics.representation == "body") & (all_fold_metrics.method == "ridge_pca32")]
            pair = token.merge(body, on="test_year", suffixes=("_token", "_body"))
            for _, row in pair.iterrows():
                token_body.append({"model": model, "prompt": prompt, "test_year": int(row.test_year),
                                   "token_rank_ic": row.rank_ic_token, "body_rank_ic": row.rank_ic_body,
                                   "rank_ic_delta": row.rank_ic_token - row.rank_ic_body})
    token_body = pd.DataFrame(token_body); token_body.to_csv(output / "token_body_paired_yearly.csv", index=False)
    paired_rows = []
    for model in MODELS:
        for prompt in PROMPTS:
            delta, low, high = paired_block_ci(frames[f"{model}__{prompt}__token"], frames[f"{model}__{prompt}__body"])
            yearly = token_body[(token_body.model == model) & (token_body.prompt == prompt)]
            paired_rows.append({"model": model, "prompt": prompt, "rank_ic_delta": delta,
                                "block_ci_low": low, "block_ci_high": high,
                                "positive_delta_years": int((yearly.rank_ic_delta > 0).sum()), "years": len(yearly)})
    pd.DataFrame(paired_rows).to_csv(output / "token_body_paired_bootstrap.csv", index=False)

    structure_rows = []
    for model in MODELS:
        for prompt in PROMPTS:
            for representation in ("token", "body"):
                for year in test_years:
                    fold = args.root / "rolling" / model / prompt / representation / str(year)
                    fixed = json.loads((fold / "fixed_parameters.json").read_text(encoding="utf-8"))
                    diagnostics = json.loads((fold / "cluster_diagnostics.json").read_text(encoding="utf-8"))
                    for method in ("hard_kmeans", "soft_kmeans", "umap_hdbscan"):
                        structure_rows.append({"model": model, "prompt": prompt, "representation": representation,
                                               "test_year": year, "method": method,
                                               "k": fixed["kmeans_k"] if method != "umap_hdbscan" else diagnostics["hdbscan_clusters_including_noise"],
                                               "ari": diagnostics.get("hard_kmeans_ari") if method != "umap_hdbscan" else None,
                                               "silhouette": diagnostics.get("hard_kmeans_silhouette") if method != "umap_hdbscan" else None,
                                               "cluster_return_order": None,
                                               "noise_fraction": diagnostics.get("hdbscan_noise_fraction") if method == "umap_hdbscan" else None})
    pd.DataFrame(structure_rows).to_csv(output / "cluster_structure_yearly.csv", index=False)

    fusion = load_fusion(args.root, test_years)
    fusion_rows, portfolio_rows = [], []
    for (group, method), frame in fusion.groupby(["group", "method"]):
        yearly = []
        for year, year_frame in frame.groupby("test_year"):
            metrics = prediction_metrics(year_frame); yearly.append(metrics["rank_ic"])
            fusion_rows.append({"group": group, "method": method, "test_year": year, **metrics})
        portfolio_rows.append({"group": group, "method": method, **portfolio(frame)})
    fusion_yearly = pd.DataFrame(fusion_rows); fusion_yearly.to_csv(output / "fusion_yearly_metrics.csv", index=False)
    fusion_summary = fusion_yearly.groupby(["group", "method"], as_index=False).agg(
        rank_ic=("rank_ic", "mean"), positive_years=("rank_ic", lambda x: int((x > 0).sum())),
        rank_ic_ir=("rank_ic_ir", "mean"), top20_return=("top20_return", "mean"),
        top20_long_short=("top20_long_short", "mean"), positive_months=("positive_months", "sum"), months=("months", "sum"),
    )
    fusion_summary.to_csv(output / "fusion_summary.csv", index=False)
    pd.DataFrame(portfolio_rows).to_csv(output / "portfolio_5bp.csv", index=False)

    qwen_ridge = fusion[(fusion.group == "model_qwen3_embedding_8b_tokens") & (fusion.method == "ridge")]
    qwen_best = fusion[(fusion.group == "model_qwen3_embedding_8b_tokens") & (fusion.method == "best_single")]
    qwen_delta, qwen_low, qwen_high = paired_block_ci(qwen_ridge, qwen_best)
    all_ridge = fusion[(fusion.group == "all_12_tokens") & (fusion.method == "ridge")]
    all_best = fusion[(fusion.group == "all_12_tokens") & (fusion.method == "best_single")]
    all_delta, all_low, all_high = paired_block_ci(all_ridge, all_best)
    qwen_year = fusion_yearly[(fusion_yearly.group == "model_qwen3_embedding_8b_tokens")].pivot(index="test_year", columns="method", values="rank_ic")
    all_year = fusion_yearly[(fusion_yearly.group == "all_12_tokens")].pivot(index="test_year", columns="method", values="rank_ic")

    geometry = pd.read_csv(args.root / "geometry" / "pairwise_token_geometry_summary.csv")
    geometry_model = geometry.groupby("model", as_index=False).agg(
        raw_cosine=("raw_cosine_mean", "mean"), centered_cosine=("centered_cosine_mean", "mean"),
        standardized_l2=("standardized_l2_mean", "mean"),
    )
    geometry_model.to_csv(output / "model_geometry_summary.csv", index=False)
    agreement_model = agreement.groupby("model", as_index=False).agg(
        factor_spearman=("spearman", "mean"), top20_overlap=("top20_overlap", "mean"),
    )
    agreement_model.to_csv(output / "model_factor_agreement_summary.csv", index=False)
    geo = geometry_model.set_index("model"); agree = agreement_model.set_index("model")
    qwen_coherent = abs(geo.loc["qwen3_embedding_8b", "centered_cosine"] - geo.loc["bge_m3", "centered_cosine"]) < abs(geo.loc["qwen3_embedding_8b", "centered_cosine"] - geo.loc["roberta", "centered_cosine"])
    stable_year_threshold = math.ceil(len(test_years) * 2 / 3)
    checks = {
        "roberta_centered_cosine_below_bge": bool(geo.loc["roberta", "centered_cosine"] < geo.loc["bge_m3", "centered_cosine"]),
        "bge_highest_factor_spearman": bool(agree.loc["bge_m3", "factor_spearman"] == agree.factor_spearman.max()),
        "bge_highest_top20_overlap": bool(agree.loc["bge_m3", "top20_overlap"] == agree.top20_overlap.max()),
        "qwen_geometry_closer_to_bge": bool(qwen_coherent),
        "qwen_factor_spearman_below_bge": bool(agree.loc["qwen3_embedding_8b", "factor_spearman"] < agree.loc["bge_m3", "factor_spearman"]),
        "qwen_fusion_delta_ci_above_zero": bool(qwen_low > 0),
        "qwen_fusion_delta_positive_years_at_least_threshold": bool(((qwen_year.ridge - qwen_year.best_single) > 0).sum() >= stable_year_threshold),
        "all_model_fusion_delta_ci_above_zero": bool(all_low > 0),
        "all_model_fusion_delta_positive_years_at_least_threshold": bool(((all_year.ridge - all_year.best_single) > 0).sum() >= stable_year_threshold),
    }
    decision = {"supported": bool(all(checks.values())), "checks": checks,
                "dataset": dataset, "test_years": test_years,
                "stable_year_threshold": stable_year_threshold,
                "qwen_fusion_rankic_delta": qwen_delta, "qwen_fusion_block_ci": [qwen_low, qwen_high],
                "all_model_fusion_rankic_delta": all_delta, "all_model_fusion_block_ci": [all_low, all_high]}
    (output / "hypothesis_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pca = base_summary[base_summary.method.eq("ridge_pca32")]
    table = pca.groupby(["model", "representation"], as_index=False).agg(rank_ic=("rank_ic", "mean"), positive_years=("positive_years", "mean"))
    dataset_zh = "新浪" if dataset == "sina" else "巨潮"
    lines = [
        f"## {dataset_zh}三模型四Prompt严格公平比较（2026-08-26）", "",
        f"本节使用{dataset_zh} `masked_short` 的{root_manifest['rows']:,}条三模型四Prompt共同新闻，按相同 `row_index` 对齐；标签为 `next_day_return`，采用{root_manifest['rolling_protocol']}严格滚动，测试年为{test_years[0]}--{test_years[-1]}。只运行训练期拟合的PCA32，不运行原始空间或PCA64回归。所有基础模型固定 `Ridge alpha=100`、`KMeans k=6`、`shrinkage=500`、`temperature=0.5`，不做逐折超参数搜索。", "",
        "### PCA32基础因子", "", "| 模型 | 表示 | 平均RankIC | 平均正年份数 |", "|---|---|---:|---:|",
    ]
    for row in table.itertuples(): lines.append(f"| {row.model} | {row.representation} | {row.rank_ic:.5f} | {row.positive_years:.2f}/{len(test_years)} |")
    lines += ["", "### 跨Prompt结构", "", "| 模型 | centered cosine | 因子Spearman | Top20重合 |", "|---|---:|---:|---:|"]
    for row in geometry_model.merge(agreement_model, on="model").itertuples():
        lines.append(f"| {row.model} | {row.centered_cosine:.5f} | {row.factor_spearman:.5f} | {row.top20_overlap:.2%} |")
    lines += ["", "### 关键假设", "", f"Qwen四token Ridge融合相对验证期最佳单token的RankIC增量为 `{qwen_delta:+.5f}`，20日日期区块bootstrap 95%区间为 `[{qwen_low:+.5f}, {qwen_high:+.5f}]`。",
              f"三模型12-token Ridge融合增量为 `{all_delta:+.5f}`，区间为 `[{all_low:+.5f}, {all_high:+.5f}]`。", "",
              f"稳定性阈值按三分之二测试年固定为 `{stable_year_threshold}/{len(test_years)}`；联合判定：**{'支持' if decision['supported'] else '不支持或仅部分支持'}**。各条件详见该数据集目录下的 `summary/hypothesis_decision.json`。", "",
              "口径限制：Qwen prompt末尾包含句号且采用后置prompt的因果模型结构；目标span不含句号。RoBERTa/BGE-M3正文使用`body_mean`，Qwen使用`article_mean`，不再使用`full_mean`冒充正文。", "",
              "既有Qwen全量收益实验仍显示token没有全面超过正文：线性RankIC `0.05360 < 0.05680`，成本后年化 `7.37% < 8.57%`；软聚类是局部例外。"]
    section = "\n".join(lines)
    (output / "report_zh.md").write_text(section + "\n", encoding="utf-8")
    update_report(args.report, section, dataset)
    (output / "COMPLETED").write_text("three_model_four_prompt_fair_summary_pca_v2\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
