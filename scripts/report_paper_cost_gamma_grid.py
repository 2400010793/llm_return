"""Build the paper-cost long, short, and long-short gamma-grid report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {
    "cluster": "BGE-M3收益token软聚类",
    "excess_return_bge_m3_short_excess_span": "BGE-M3超额token非mask",
    "excess_return_roberta_masked_short_excess_span": "RoBERTa超额token mask",
    "loss_roberta_masked_short_loss_span": "RoBERTa亏损token mask",
    "pooled_mean_logistic": "平均池化Logistic",
    "profit_roberta_masked_short_profit_span": "RoBERTa盈利token mask",
    "profit_roberta_short_profit_span": "RoBERTa盈利token非mask",
    "return_roberta_masked_short_plain_return_span": "RoBERTa收益token mask",
    "return_token_logistic": "收益token Logistic",
}
MODE_LABELS = {
    "long_only": "仅做多",
    "short_only": "仅做空（理论）",
    "long_short": "多空（理论）",
}


def percent(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def number(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fixed_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["候选"] = result["candidate"].map(LABELS)
    result["组合"] = result["mode"].map(MODE_LABELS)
    result["毛Sharpe"] = result["gross_sharpe"].map(number)
    result["净日均(bp)"] = result["net_mean_bps"].map(lambda x: number(x, 3))
    result["净算术年化"] = result["net_annual_return"].map(percent)
    result["净复利年化"] = result["net_geometric_annual_return"].map(percent)
    result["全期净累计"] = result["net_cumulative_return"].map(percent)
    result["净Sharpe"] = result["net_sharpe"].map(number)
    result["日均换手"] = result["turnover"].map(percent)
    result["日均成本(bp)"] = result["cost_bps_per_day"].map(lambda x: number(x, 3))
    return result[[
        "候选", "组合", "毛Sharpe", "净日均(bp)", "净算术年化",
        "净复利年化", "全期净累计", "净Sharpe", "日均换手", "日均成本(bp)",
    ]]


def matrix(frame: pd.DataFrame, column: str, *, as_percent: bool) -> pd.DataFrame:
    pivot = frame.pivot(index="candidate", columns="gamma", values=column)
    pivot = pivot.reindex(LABELS)
    pivot.index = pivot.index.map(LABELS)
    pivot.columns = [f"g={value:.1f}" for value in pivot.columns]
    if as_percent:
        pivot = pivot.apply(lambda col: col.map(percent))
    else:
        pivot = pivot.apply(lambda col: col.map(number))
    return pivot.reset_index(names="候选")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_runs = pd.read_csv(args.input)
    paper = all_runs.loc[all_runs["cost"].eq("paper_10bp_fallback")].copy()
    expected = len(LABELS) * len(MODE_LABELS) * 10
    if len(paper) != expected:
        raise ValueError(f"expected {expected} paper runs, found {len(paper)}")
    if set(paper["candidate"]) != set(LABELS):
        raise ValueError("paper grid candidate set is incomplete")
    if set(paper["mode"]) != set(MODE_LABELS):
        raise ValueError("paper grid portfolio mode set is incomplete")
    expected_gammas = np.round(np.arange(0.1, 1.01, 0.1), 1)
    for key, group in paper.groupby(["candidate", "mode"]):
        actual = np.sort(group["gamma"].round(1).unique())
        if not np.array_equal(actual, expected_gammas):
            raise ValueError(f"incomplete gamma grid for {key}: {actual}")

    fixed = paper.loc[np.isclose(paper["gamma"], 0.1)].copy()
    best = (
        paper.sort_values("net_sharpe", ascending=False)
        .groupby(["candidate", "mode"], as_index=False)
        .first()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paper.sort_values(["mode", "candidate", "gamma"]).to_csv(
        args.output_dir / "paper_all_gamma_results.csv", index=False
    )
    fixed.sort_values(["mode", "net_sharpe"], ascending=[True, False]).to_csv(
        args.output_dir / "paper_fixed_gamma_0p10.csv", index=False
    )
    best.sort_values(["mode", "net_sharpe"], ascending=[True, False]).to_csv(
        args.output_dir / "paper_ex_post_best_gamma.csv", index=False
    )

    lines = [
        "# 论文成本下做多、做空与多空 Gamma 全网格",
        "",
        "## 口径",
        "",
        "- 样本：新浪股票日预测，2018--2026，共 2,082 个交易日。",
        "- 组合：Q5仅做多、Q1仅做空、Q5-Q1多空；EWCT gamma=0.1--1.0。",
        "- 成本：论文每笔大盘10bp、小盘20bp。当前市场面板没有历史市值/大小盘字段，",
        "  因此所有股票实际按每笔10bp计算，是论文成本的乐观下界，不是严格10/20bp复现。",
        "- 做空与多空没有逐日融券可用性、借券费和强平约束，属于理论组合。",
        "- 固定gamma=0.1是可比较主表；按全期测试净Sharpe挑选的最佳gamma仅作事后敏感性。",
        "",
        "## 固定 gamma=0.1",
        "",
    ]
    for mode in MODE_LABELS:
        subset = fixed.loc[fixed["mode"].eq(mode)].sort_values(
            "net_sharpe", ascending=False
        )
        lines.extend([f"### {MODE_LABELS[mode]}", "", table(fixed_table(subset)), ""])

    best_display = best.copy()
    best_display["gamma"] = best_display["gamma"].map(lambda x: f"{x:.1f}")
    lines.extend(["## 每个候选的事后最佳 gamma", ""])
    for mode in MODE_LABELS:
        subset = best_display.loc[best_display["mode"].eq(mode)].sort_values(
            "net_sharpe", ascending=False
        )
        formatted = fixed_table(subset)
        formatted.insert(2, "最佳gamma", subset["gamma"].to_numpy())
        lines.extend([f"### {MODE_LABELS[mode]}", "", table(formatted), ""])

    for mode in MODE_LABELS:
        subset = paper.loc[paper["mode"].eq(mode)]
        lines.extend([
            f"## {MODE_LABELS[mode]}：不同 gamma 的净 Sharpe",
            "",
            table(matrix(subset, "net_sharpe", as_percent=False)),
            "",
            f"## {MODE_LABELS[mode]}：不同 gamma 的净复利年化",
            "",
            table(matrix(subset, "net_geometric_annual_return", as_percent=True)),
            "",
        ])

    fixed_winners = fixed.loc[fixed.groupby("mode")["net_sharpe"].idxmax()]
    conclusions = []
    for row in fixed_winners.itertuples(index=False):
        conclusions.append(
            f"- {MODE_LABELS[row.mode]}：固定gamma=0.1最好的是"
            f"{LABELS[row.candidate]}，净复利年化{percent(row.net_geometric_annual_return)}，"
            f"净Sharpe {number(row.net_sharpe)}。"
        )
    lines.extend([
        "## 结论",
        "",
        *conclusions,
        "- gamma 增大时换手和成本快速上升；任何事后最优 gamma 都不能直接作为正式样本外结论。",
        "- 做空与多空结果在加入实际融券限制、借券费及严格10/20bp大小盘成本后只会更弱。",
        "",
        "## 产物",
        "",
        "- `paper_all_gamma_results.csv`：270个完整run。",
        "- `paper_fixed_gamma_0p10.csv`：固定gamma主表。",
        "- `paper_ex_post_best_gamma.csv`：事后最优gamma敏感性表。",
    ])
    (args.output_dir / "paper_gamma_long_short_report_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print({"paper_runs": len(paper), "fixed": len(fixed), "best": len(best)})


if __name__ == "__main__":
    main()
