"""Collect Qwen simple_states and cost-backtest outputs into stable tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def number(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "NA"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    simple = []
    for path in sorted((args.root / "simple_states").glob("*/*.stats.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "protocol", path.parent.name)
        simple.append(frame)
    simple_frame = (pd.concat(simple, ignore_index=True) if simple else pd.DataFrame(
        [{"status": "missing_or_failed"}]
    ))
    simple_frame.to_csv(args.root / "simple_states_metrics.csv", index=False)

    portfolio = []
    for path in sorted((args.root / "portfolio").glob("*/*/*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        parts = path.relative_to(args.root / "portfolio").parts
        protocol, cost, side = parts[:3]
        for gamma, run in payload.get("runs", {}).items():
            metrics = run.get("metrics", {})
            flattened = {}
            for group, values in metrics.items():
                if isinstance(values, dict):
                    flattened.update({f"{group}_{key}": value for key, value in values.items()})
                else:
                    flattened[group] = values
            portfolio.append({"protocol": protocol, "cost": cost, "side": side,
                              "gamma": gamma, **flattened})
    portfolio_frame = pd.DataFrame(portfolio)
    portfolio_frame.to_csv(args.root / "portfolio_cost_metrics.csv", index=False)

    selected = json.loads((args.root / "selected_candidates.json").read_text(encoding="utf-8"))["selected"]
    best = max((row for row in selected if row["model"] == "qwen3_embedding_8b"),
               key=lambda row: (float(row["rank_ic_mean"]), float(row["oos_r2_vs_historical_mean"]), -float(row["mae"])))
    strict = pd.read_csv(args.root / "strict_2026_metrics.csv")
    rolling = pd.read_csv(args.root / "rolling_3_1_1_yearly_metrics.csv")
    match = lambda frame: frame[(frame.model == best["model"]) & (frame.variant == best["variant"]) &
                                (frame.regressor == best["regressor"]) & (frame.reducer == best["reducer"]) &
                                (frame.components.astype(int) == int(best["components"])) &
                                (frame.candidate_role == "best")]
    strict_best, rolling_best = match(strict), match(rolling)
    paired = pd.read_csv(args.root / "paired_model_comparison.csv")
    absolute = paired[paired.left_model == "zero_ic"].iloc[0]
    positive_years = int((rolling_best.rank_ic_mean > 0).sum())
    monthly = pd.read_csv(args.root / "monthly_rankic.csv")
    positive_months = int((monthly.rank_ic > 0).sum())
    cluster = pd.read_csv(args.root / "cluster_linear_increment.csv")
    cluster_positive = int((cluster.rank_ic_increment > 0).sum())
    strict_rank_ic = strict_best.rank_ic_mean.iloc[0] if len(strict_best) else float("nan")
    plain_spec = next(row for row in selected if row["model"] == "qwen3_embedding_8b" and row["variant"] == "plain")
    plain = strict[(strict.model == plain_spec["model"]) & (strict.variant == "plain") &
                   (strict.regressor == plain_spec["regressor"]) &
                   (strict.reducer == plain_spec["reducer"]) &
                   (strict.components.astype(int) == int(plain_spec["components"])) &
                   (strict.candidate_role == "best")]
    plain_rank_ic = plain.rank_ic_mean.iloc[0] if len(plain) else float("nan")
    baseline_models = strict[(strict.model.isin(["roberta", "bge_m3"])) &
                             (strict.candidate_role == "best")]
    baseline_best = float(baseline_models.rank_ic_mean.max()) if len(baseline_models) else float("nan")
    paper_ls = portfolio_frame[(portfolio_frame.protocol == "strict_2026") &
                               (portfolio_frame.cost == "paper") &
                               (portfolio_frame.side == "long_short")]
    net_sharpe = float(paper_ls.net_sharpe.iloc[0]) if len(paper_ls) and "net_sharpe" in paper_ls else float("nan")
    huber = pd.read_csv(args.root / "huber_seed_stability.csv")
    huber = huber[huber.variant == best["variant"]]
    huber_no_flip = bool((huber.mean_rank_ic > 0).all() or (huber.mean_rank_ic < 0).all()) if len(huber) == 5 else False
    strong = {
        "strict_rank_ic_positive": bool(strict_rank_ic > 0),
        "strict_bootstrap_ci_positive": bool(float(absolute.ci_low) > 0),
        "rolling_positive_years_at_least_4": positive_years >= 4,
        "positive_months_majority": positive_months > len(monthly) / 2,
        "prompt_or_mask_beats_plain": bool(best["variant"] != "plain" and strict_rank_ic > plain_rank_ic),
        "not_weaker_than_roberta_bge": bool(strict_rank_ic >= baseline_best),
        "paper_cost_net_long_short_sharpe_positive": bool(net_sharpe > 0),
        "huber_seeds_no_direction_flip": huber_no_flip,
    }
    report = [
        "# Qwen 2018--2026 线性收益预测报告", "",
        "## 冻结配置", "",
        f"- 表示：`{best['variant']}`；模型：`{best['regressor']}`；降维：`{best['reducer']}{best['components']}`。",
        "- 该配置仅由2018--2023训练、2024--2025验证选择，2026未参与选择。", "",
        "## 样本外结果", "",
        f"- 2026严格测试RankIC：{number(strict_rank_ic)}。",
        f"- 2026 RankIC 20日区块bootstrap 95%区间：[{number(absolute.ci_low)}, {number(absolute.ci_high)}]。",
        f"- 3/1/1辅助滚动正RankIC年份：{positive_years}/5。",
        f"- 正RankIC月份：{positive_months}/{len(monthly)}。",
        f"- 聚类增强RankIC为正的测试fold：{cluster_positive}/{len(cluster)}。", "",
        f"- Qwen plain严格测试RankIC：{number(plain_rank_ic)}；RoBERTa/BGE最佳值：{number(baseline_best)}。",
        f"- 论文成本严格测试净多空Sharpe：{number(net_sharpe)}。", "",
        "## 强预测能力判定", "",
        *[f"- {'通过' if value else '未通过'}：`{key}`" for key, value in strong.items()], "",
        "只有全部核心条件通过，才能称为强预测能力；否则结论为弱信号或证据不足。", "",
        "## 回测口径", "",
        "- `simple_states_metrics.csv`为平台结果。",
        "- `portfolio_cost_metrics.csv`分别包含论文成本和A股单边5bp成本下的多空、做多、做空结果。",
        "- 训练目标为市场残差，持仓回测实际收益已替换为原始可交易O2O收益。",
    ]
    (args.root / "report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
