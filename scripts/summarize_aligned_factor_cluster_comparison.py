#!/usr/bin/env python3
"""Summarize token-vs-body clustering results for the new prompt axes."""
from pathlib import Path
import json
import pandas as pd


def main() -> None:
    root = Path("/data/alpha_team2/shares/llm_return/reports/aligned_factor_clusters")
    files = sorted(root.glob("*_return3d.csv"))
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    wide = data.pivot_table(
        index=["axis", "method", "target", "test_year"],
        columns="representation", values=["rankic", "top20_ls"], aggfunc="first",
    )
    for metric in ("rankic", "top20_ls"):
        wide[(metric, "delta_token_minus_body")] = wide[(metric, "token")] - wide[(metric, "body")]
    delta = wide[[('rankic', 'delta_token_minus_body'), ('top20_ls', 'delta_token_minus_body')]].reset_index()
    delta.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in delta.columns]
    delta = delta.rename(columns={
        "rankic_delta_token_minus_body": "rankic_delta_token_body",
        "top20_ls_delta_token_minus_body": "top20_ls_delta_token_body",
    })
    summary = (
        delta.groupby("method", as_index=False)
        .agg(
            axes=("axis", "nunique"),
            mean_rankic_delta=("rankic_delta_token_body", "mean"),
            median_rankic_delta=("rankic_delta_token_body", "median"),
            rankic_wins=("rankic_delta_token_body", lambda s: int((s > 0).sum())),
            mean_top20_delta=("top20_ls_delta_token_body", "mean"),
            top20_wins=("top20_ls_delta_token_body", lambda s: int((s > 0).sum())),
        )
    )
    axis_summary = (
        delta.groupby(["axis", "method"], as_index=False)
        .agg(
            token_rankic=("rankic_delta_token_body", "mean"),
            token_top20=("top20_ls_delta_token_body", "mean"),
        )
    )
    data.to_csv(root / "all_results.csv", index=False)
    delta.to_csv(root / "token_minus_body_delta.csv", index=False)
    summary.to_csv(root / "method_summary.csv", index=False)
    axis_summary.to_csv(root / "axis_method_summary.csv", index=False)
    payload = {
        "protocol": "2026 test; 2018-2023 train; 2024-2025 validation; target=forward_compounded_return_3d",
        "rows": int(len(data)),
        "summary": summary.to_dict(orient="records"),
        "axis_summary": axis_summary.to_dict(orient="records"),
    }
    (root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 新 Prompt Token Embedding 聚类收益预测",
        "",
        "本实验使用当前 18 个新 prompt 的 RoBERTa masked-short token-span direction embedding，并与同轴 body_mean direction embedding 严格配对。测试窗口为 2026，训练 2018--2023，验证 2024--2025，标签为 `forward_compounded_return_3d`。",
        "",
        "## 方法",
        "",
        "- PCA+Ridge",
        "- PCA+MiniBatchKMeans 硬簇特征+Ridge",
        "- PCA+距离温度软 KMeans 连续分数",
        "- PCA64+UMAP8+HDBSCAN+Ridge",
        "",
        "## Token 相对 Body",
        "",
        "| 方法 | 平均 RankIC 增量 | 中位 RankIC 增量 | RankIC 胜出轴数 | 平均 Top20 多空增量 | Top20 胜出轴数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['mean_rankic_delta']:+.5f} | {row['median_rankic_delta']:+.5f} | {row['rankic_wins']}/{row['axes']} | {row['mean_top20_delta']:+.5f} | {row['top20_wins']}/{row['axes']} |")
    lines += [
        "",
        "## 轴级结果",
        "",
        "| 轴 | 方法 | Token-Body RankIC | Token-Body Top20 多空 |",
        "|---|---|---:|---:|",
    ]
    for row in axis_summary.to_dict(orient="records"):
        lines.append(f"| {row['axis']} | {row['method']} | {row['token_rankic']:+.5f} | {row['token_top20']:+.5f} |")
    lines += [
        "",
        "## 结论边界",
        "",
        "如果 token embedding 只是无关噪声，它在相同 PCA/聚类流程下不应系统性优于 body embedding。当前结果需要同时看方法和轴：token 在部分轴的 PCA/硬聚类上有明显增量，但软聚类和 UMAP 并不稳定。因此这支持‘新 token 改变了可预测表示’，但还不能宣称所有新 prompt token 都普遍优于正文。",
        "",
        "这是一年严格样本外窗口，不能替代至少 6/9 年的稳定性检验。下一步应对每个轴使用匹配标签（估值、波动率、冲击、流动性）再重复同一聚类流程，并报告区块 bootstrap。",
    ]
    (root / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
