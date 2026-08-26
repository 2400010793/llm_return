"""Audit and compare existing OOS body/prompt word-span predictions.

This intentionally consumes already-generated stock-day predictions.  It does
not retrain models or treat Qwen article_mean as an exact body_mean; the alias
is recorded in the audit and report outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROMPTS = {
    "profit": "盈利",
    "excess_return": "超额收益",
    "return": "收益",
    "loss": "亏损",
}
SPAN_REPRS = {
    "profit": "profit_span",
    "excess_return": "excess_span",
    "return": "plain_return_span",
    "loss": "loss_span",
}


def rank_ic(frame: pd.DataFrame, score: str) -> float:
    values = []
    for _, g in frame.groupby("entry_date", sort=False):
        if len(g) >= 5 and g[score].nunique() > 1 and g.actual_return.nunique() > 1:
            values.append(g[score].corr(g.actual_return, method="spearman"))
    return float(np.nanmean(values)) if values else np.nan


def top20(frame: pd.DataFrame, score: str) -> dict[str, float]:
    rows = []
    for _, g in frame.groupby("entry_date", sort=False):
        g = g.dropna(subset=[score, "actual_return"])
        if len(g) < 10:
            continue
        n = max(1, int(np.ceil(len(g) * 0.2)))
        s = g.sort_values([score, "stock_id"], kind="mergesort")
        long = float(s.tail(n).actual_return.mean())
        low = float(s.head(n).actual_return.mean())
        rows.append((long, low, long - low))
    if not rows:
        return {"long_bp": np.nan, "short_leg_bp": np.nan, "ls_bp": np.nan, "days": 0}
    a = np.asarray(rows, dtype=float)
    return {"long_bp": float(a[:, 0].mean() * 1e4), "short_leg_bp": float(a[:, 1].mean() * 1e4), "ls_bp": float(a[:, 2].mean() * 1e4), "days": int(len(a))}


def load(path: Path, model: str, prompt: str, variant: str, representation: str, alias: str) -> tuple[pd.DataFrame, dict]:
    d = pd.read_parquet(path, columns=["stock_id", "entry_date", "actual_return", "prediction"])
    d["entry_date"] = pd.to_datetime(d["entry_date"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["entry_date", "actual_return", "prediction"]).drop_duplicates(["stock_id", "entry_date"])
    d["test_year"] = d.entry_date.dt.year
    name = f"{model}__{prompt}__{variant}__{representation}"
    d = d.rename(columns={"prediction": name})
    return d, {"factor": name, "model": model, "prompt": prompt, "variant": variant, "representation": representation, "representation_alias": alias, "path": str(path), "rows": int(len(d)), "years": sorted(map(int, d.test_year.unique()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sina-root", type=Path, required=True)
    ap.add_argument("--qwen-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--variant", default="masked_short", choices=["short", "masked_short"])
    ap.add_argument("--mechanism-root", type=Path, default=None, help="Existing prompt mechanism summary root")
    args = ap.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    factors: list[pd.DataFrame] = []
    audits: list[dict] = []
    missing: list[dict] = []

    # RoBERTa/BGE-M3: existing prompt-specific full_mean and word spans.
    for model in ("roberta", "bge_m3"):
        for prompt, word in PROMPTS.items():
            for rep, directory in [("body_mean", "full_mean"), ("word_span", SPAN_REPRS[prompt])]:
                path = args.sina_root / f"{prompt}_{model}_{args.variant}_{directory}" / "stock_day_predictions.stock_day_predictions.parquet"
                if not path.exists():
                    missing.append({"model": model, "prompt": prompt, "variant": args.variant, "representation": rep, "expected": str(path)})
                    continue
                d, audit = load(path, model, prompt, args.variant, rep, "full_mean" if rep == "body_mean" else directory)
                factors.append(d); audits.append(audit)

    # Qwen current completed prompt: article_mean is explicitly a body proxy.
    qbase = args.qwen_root / "results" / f"{args.variant}_article_mean" / "stock_day_predictions.stock_day_predictions.parquet"
    qspan = args.qwen_root / "results" / f"{args.variant}_return_token" / "stock_day_predictions.stock_day_predictions.parquet"
    for path, rep, alias in [(qbase, "body_mean", "article_mean_body_proxy"), (qspan, "word_span", "return_token")]:
        if not path.exists():
            missing.append({"model": "qwen3_embedding_8b", "prompt": "return", "variant": args.variant, "representation": rep, "expected": str(path)})
            continue
        d, audit = load(path, "qwen3_embedding_8b", "return", args.variant, rep, alias)
        factors.append(d); audits.append(audit)

    if not factors:
        raise SystemExit("no completed prediction files found")
    # Common stock-day intersection is used for correlation and model ranking.
    merged = factors[0]
    for d in factors[1:]:
        merged = merged.merge(d, on=["stock_id", "entry_date", "test_year"], how="inner", validate="one_to_one", suffixes=("", "_other"))
        if "actual_return_other" in merged:
            if not np.allclose(merged.actual_return, merged.actual_return_other, equal_nan=True):
                raise ValueError("actual_return mismatch across factor files")
            merged = merged.drop(columns=["actual_return_other"])
    factor_names = [c for c in merged.columns if "__" in c and c not in {"stock_id", "entry_date"}]
    yearly, monthly = [], []
    for name in factor_names:
        meta = next(x for x in audits if x["factor"] == name)
        for year, g in merged.groupby("test_year"):
            m = top20(g, name)
            yearly.append({**meta, "test_year": int(year), "rank_ic": rank_ic(g, name), **m, "rows": len(g)})
        for month, g in merged.groupby(merged.entry_date.dt.to_period("M")):
            m = top20(g, name)
            monthly.append({**meta, "month": str(month), "rank_ic": rank_ic(g, name), **m, "rows": len(g)})
    corr = merged[factor_names].corr(method="spearman")
    corr.to_csv(out / "factor_spearman_correlation.csv")
    if args.mechanism_root:
        token_path = args.mechanism_root / "summary" / "token_metrics_all.parquet"
        agreement_path = args.mechanism_root / "summary" / "cross_model_agreement.csv"
        if token_path.exists():
            token = pd.read_parquet(token_path)
            token["model"] = token["model"].replace({"qwen3_embedding_8b": "qwen3_embedding_8b"})
            token_summary = token.groupby(["model", "prompt_length", "variant", "target", "semantic_group"], as_index=False).agg(
                mean_cosine_distance=("context_cosine_distance", "mean"),
                mean_standardized_l2=("context_standardized_l2", "mean"),
                mean_mask_delta_l2=("mask_delta_l2", "mean"),
                mean_fisher=("fisher", "mean"),
                median_fisher_rank=("fisher_rank", "median"),
                rows=("token", "size"),
            )
            token_summary.to_csv(out / "token_cosine_summary.csv", index=False)
        if agreement_path.exists():
            pd.read_csv(agreement_path).to_csv(out / "cross_model_token_agreement.csv", index=False)
    pd.DataFrame(yearly).to_csv(out / "model_yearly_metrics.csv", index=False)
    pd.DataFrame(monthly).to_csv(out / "model_monthly_metrics.csv", index=False)
    pd.DataFrame(audits).to_json(out / "alignment_audit.json", orient="records", force_ascii=False, indent=2)
    pd.DataFrame(missing).to_csv(out / "missing_combinations.csv", index=False)
    summary = pd.DataFrame(yearly).groupby(["model", "prompt", "representation", "representation_alias"], as_index=False).agg(
        test_years=("test_year", "nunique"), mean_rank_ic=("rank_ic", "mean"), positive_years=("rank_ic", lambda x: int((x > 0).sum())),
        mean_long_bp=("long_bp", "mean"), mean_short_leg_bp=("short_leg_bp", "mean"), mean_ls_bp=("ls_bp", "mean"),
    )
    summary.to_csv(out / "model_factor_comparison.csv", index=False)
    report = ["# 三模型 Prompt Body/Word-Span 首轮审计", "", "本报告只使用已有样本外股票日预测，不重新训练。RoBERTa/BGE-M3 的 `full_mean` 标为 body_mean；Qwen 的 `article_mean` 标为 body_mean proxy。", "", "## 可用因子", "", summary.to_markdown(index=False), "", "## 缺失组合", "", pd.DataFrame(missing).to_markdown(index=False) if missing else "无", "", "## 口径限制", "", "- 当前结果是已有 OOS 预测的描述性和配对比较；无 prompt body baseline 若缺失，不报告为 prompt 因果增量。", "- Qwen 当前仅有 `分析股票收益` 的 article_mean 与收益 token，不能冒充四 prompt 全量实验。", "- 正式结论需在共同股票日上完成 2018--2026 重新滚动训练、bootstrap 和 simple_states。"]
    (out / "REPORT_ALL_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"common_rows={len(merged)} factors={len(factor_names)} missing={len(missing)} output={out}")


if __name__ == "__main__":
    main()
