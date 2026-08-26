"""Summarize direction-token soft clustering and both evaluation protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score


LABELS = {
    "profit": "盈利", "excess_return": "超额收益",
    "return": "收益", "loss": "亏损",
}


def markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) if not frame.empty else "无结果"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    for path in sorted((args.root / "folds").rglob("metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for objective, selected in report["selected"].items():
            selected_rows.append({
                "prompt": report["prompt"], "model": report["model"],
                "variant": report["variant"], "test_year": report["test_year"],
                "representation": report.get("representation"),
                "projection_mode": report.get("projection_mode", "pca"),
                "objective": objective, **selected,
            })
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(args.root / "selected_hyperparameters_yearly.csv", index=False)
    structure = (
        selected.groupby(["prompt", "model", "variant", "representation", "projection_mode", "objective"], dropna=False)
        .agg(mean_k=("k", "mean"), median_k=("k", "median"), mean_ari=("ari", "mean"),
             min_ari=("ari", "min"), test_years=("test_year", "nunique"))
        .reset_index()
        if not selected.empty else selected
    )
    structure.to_csv(args.root / "cluster_structure_summary.csv", index=False)

    direction_rows = []
    for path in sorted((args.root / "combined").rglob("stock_day_predictions.parquet")):
        frame = pd.read_parquet(path)
        actual = pd.to_numeric(frame["actual_return"], errors="coerce")
        for column in sorted(name for name in frame if name.startswith("prediction_")):
            prediction = pd.to_numeric(frame[column], errors="coerce")
            keep = actual.notna() & prediction.notna()
            observed = actual.loc[keep]
            score = prediction.loc[keep]
            actual_up = observed.gt(0)
            predicted_up = score.gt(0)
            ranks = score.rank(method="first", pct=True)
            direction_rows.append({
                "config": str(path.parent.relative_to(args.root / "combined")),
                "objective": column.removeprefix("prediction_"), "rows": int(keep.sum()),
                "direction_accuracy": float(predicted_up.eq(actual_up).mean()),
                "balanced_accuracy": float(balanced_accuracy_score(actual_up, predicted_up)),
                "predicted_up_rate": float(predicted_up.mean()),
                "actual_up_rate": float(actual_up.mean()),
                "top20_positive_rate": float(observed.loc[ranks.gt(0.8)].gt(0).mean()),
                "bottom20_positive_rate": float(observed.loc[ranks.le(0.2)].gt(0).mean()),
            })
    direction = pd.DataFrame(direction_rows)
    direction.to_csv(args.root / "direction_classification_metrics.csv", index=False)

    rankic_rows = []
    for path in sorted((args.root / "combined").rglob("stock_day_predictions.parquet")):
        frame = pd.read_parquet(path)
        frame["entry_date"] = pd.to_datetime(frame["entry_date"])
        config = str(path.parent.relative_to(args.root / "combined"))
        groups = list(frame.groupby("entry_date"))
        for column in sorted(name for name in frame if name.startswith("prediction_")):
            observations = []
            for date, group in groups:
                if len(group) < 5 or group[column].nunique() < 2:
                    continue
                value = spearmanr(group[column], group["actual_return"]).statistic
                if np.isfinite(value):
                    observations.append((date, float(value)))
            daily = pd.DataFrame(observations, columns=["date", "rank_ic"])
            daily["year"] = daily["date"].dt.year
            daily["month"] = daily["date"].dt.to_period("M")
            yearly = daily.groupby("year")["rank_ic"].mean()
            monthly = daily.groupby("month")["rank_ic"].mean()
            rankic_rows.append({
                "config": config, "objective": column.removeprefix("prediction_"),
                "daily_rank_ic": float(daily["rank_ic"].mean()),
                "positive_years": int(yearly.gt(0).sum()), "test_years": int(len(yearly)),
                "positive_months": int(monthly.gt(0).sum()), "test_months": int(len(monthly)),
            })
    rankic = pd.DataFrame(rankic_rows)
    rankic.to_csv(args.root / "oos_rankic_stability.csv", index=False)

    simple_rows = []
    for path in sorted((args.root / "simple_states").glob("*/*.stats.csv")):
        frame = pd.read_csv(path)
        if len(frame):
            row = frame.iloc[-1].to_dict()
            parts = path.parent.name.removeprefix("soft_").split("_")
            row["factor_id"] = path.parent.name
            row["path"] = str(path)
            simple_rows.append(row)
    simple = pd.DataFrame(simple_rows)
    simple.to_csv(args.root / "simple_states_all.csv", index=False)

    portfolio_rows = []
    for path in sorted((args.root / "portfolio").glob("*/*/*/gamma_*/summary.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        factor_id, cost, mode, gamma_dir = path.parts[-5:-1]
        metrics = report.get("metrics", {})
        portfolio_rows.append({
            "factor_id": factor_id, "cost": cost, "mode": mode,
            "gamma": float(gamma_dir.removeprefix("gamma_").replace("p", ".")),
            "gross_annual_return": metrics.get("gross", {}).get("annualized_return"),
            "gross_sharpe": metrics.get("gross", {}).get("sharpe"),
            "net_annual_return": metrics.get("net", {}).get("annualized_return"),
            "net_geometric_annual_return": metrics.get("net", {}).get("geometric_annualized_return"),
            "net_sharpe": metrics.get("net", {}).get("sharpe"),
            "net_cumulative_return": metrics.get("net", {}).get("cumulative_return"),
            "max_drawdown": metrics.get("net", {}).get("max_drawdown"),
            "turnover": metrics.get("execution", {}).get("mean_daily_turnover"),
            "transaction_cost": metrics.get("execution", {}).get("mean_daily_transaction_cost"),
            "path": str(path),
        })
    portfolio = pd.DataFrame(portfolio_rows)
    portfolio.to_csv(args.root / "portfolio_all_gamma.csv", index=False)
    fixed = portfolio.loc[portfolio["gamma"].eq(0.1)].copy() if not portfolio.empty else portfolio
    fixed.to_csv(args.root / "portfolio_fixed_gamma_0p10.csv", index=False)
    best = (
        portfolio.sort_values("net_sharpe", ascending=False)
        .groupby(["factor_id", "cost", "mode"], as_index=False).first()
        if not portfolio.empty else portfolio
    )
    best.to_csv(args.root / "portfolio_ex_post_best_gamma.csv", index=False)

    fixed_show = fixed.sort_values(["cost", "mode", "net_sharpe"], ascending=[True, True, False]).copy()
    columns = ["factor_id", "cost", "mode", "net_geometric_annual_return", "net_sharpe", "max_drawdown", "turnover"]
    if not fixed_show.empty:
        fixed_show = fixed_show[columns]
        for column in ("net_geometric_annual_return", "max_drawdown", "turnover"):
            fixed_show[column] = fixed_show[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "NA")
        fixed_show["net_sharpe"] = fixed_show["net_sharpe"].map(lambda value: f"{value:.3f}" if pd.notna(value) else "NA")
    simple_show = simple[[column for column in ("factor", "IC", "ICIR", "Ret", "RetL", "RetS", "Sharpe", "ErrorCode") if column in simple]].copy()
    lines = [
        "# 四个 Prompt 方向词 Token 软聚类评估", "",
        "- Token：盈利、超额收益、收益、亏损；每个 Prompt 独立训练。",
        "- 三个选择目标：验证期 RankIC、验证期 Q5-Q1 价差 Sharpe、验证期Top20%相对当日股票池均值的多头Sharpe。",
        "- 候选响应：原始收益、日内去均值收益、日内截面 rank；测试年不参与选择。",
        "- 正式持仓同时报告做多、独立做空和多空；固定 gamma=0.1 为主表。",
        "- paper_10bp 是缺少市值时统一10bp的论文成本乐观下界。", "",
        "## simple_states", "", markdown(simple_show), "",
        "## 固定 gamma=0.1 的持仓结果", "", markdown(fixed_show), "",
        "## 解释边界", "",
        "`RetS>0` 仅表示低分组跑输基准。只有 `short_only` 的净收益和净 Sharpe 为正，才说明独立做空盈利。",
        "按测试期挑出的最佳 gamma 只保存在敏感性表，不作为正式样本外结论。", "",
    ]
    (args.root / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    audit = {"fold_selections": len(selected), "simple_states": len(simple), "portfolio_runs": len(portfolio), "direction_rows": len(direction), "rankic_rows": len(rankic)}
    (args.root / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
