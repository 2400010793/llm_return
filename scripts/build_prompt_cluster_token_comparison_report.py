#!/usr/bin/env python3
"""Build a reproducible comparison of PCA/cluster methods and prompt tokens.

The source experiments are already complete under the Sina prompt_positive_v1
tree.  This script only reads their frozen summaries and derives paired deltas;
it never selects a test-year model or changes any predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(
    "/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026"
    "/single_stock_cninfo_v1/prompt_positive_v1"
)
OUT = Path("/mnt/lustre3/home/gaozh/llm_return/reports")
PROMPTS = ["profit", "excess_return", "return", "loss"]
PROMPT_ZH = {
    "profit": "盈利",
    "excess_return": "超额收益",
    "return": "收益",
    "loss": "亏损",
}
METHOD_FILES = {
    "linear_ridge": "prompt_late_fusion_{model}_{variant}_word_span_yearly_overall.csv",
    "hard_kmeans_ridge": "prompt_cluster_late_fusion_{model}_{variant}_word_span_yearly_overall.csv",
}
MODEL_VARIANTS = [
    ("roberta", "short"),
    ("roberta", "masked_short"),
    ("bge_m3", "short"),
    ("bge_m3", "masked_short"),
]


def linear_hard() -> pd.DataFrame:
    rows = []
    for family in METHOD_FILES:
        for model, variant in MODEL_VARIANTS:
            path = BASE / "summary" / METHOD_FILES[family].format(model=model, variant=variant)
            x = pd.read_csv(path).rename(columns={"method": "prompt"})
            x = x[x["prompt"].isin(PROMPTS)].copy()
            x["method_family"] = family
            x["model"] = model
            x["variant"] = variant
            rows.append(x[["prompt", "model", "variant", "rankic", "long_net", "ls_net", "positive_years", "method_family"]])
    data = pd.concat(rows, ignore_index=True)
    data["prompt_zh"] = data["prompt"].map(PROMPT_ZH)
    return data


def soft_cluster() -> pd.DataFrame:
    path = BASE / "soft_direction_tokens_v1" / "selected_hyperparameters_yearly.csv"
    x = pd.read_csv(path)
    # daily_rank_ic is the test-year stock-day ranking metric retained by the
    # fold runner; aggregate only after each fixed protocol is complete.  It is
    # intentionally kept separate from the report's simple_states IC, which is
    # computed on a different platform protocol.
    out = (
        x.groupby(["prompt", "model", "variant", "objective"], as_index=False)
        .agg(
            rankic=("daily_rank_ic", "mean"),
            positive_years=("daily_rank_ic", lambda s: int((s > 0).sum())),
            spread_mean=("spread_mean", "mean"),
            spread_sharpe=("spread_sharpe", "mean"),
            mean_ari=("ari", "mean"),
            mean_k=("k", "mean"),
        )
    )
    out["prompt_zh"] = out["prompt"].map(PROMPT_ZH)
    out["method_family"] = "soft_kmeans_distance_shrinkage"
    return out


def umap_hdbscan() -> pd.DataFrame:
    rows = []
    for version in ["results_pca_hdbscan_v2", "results_pca_hdbscan_fixed_v1"]:
        root = BASE / version / "bge_m3_masked_short_loss_span"
        for path in sorted(root.glob("*/metrics.json")):
            x = json.loads(path.read_text())
            test = x.get("test_top20", {})
            if not {"pca_ridge", "umap_hdbscan"}.issubset(test):
                continue
            rows.append(
                {
                    "version": version,
                    "test_year": x["test_year"],
                    "pca_ridge_top20": test["pca_ridge"],
                    "umap_hdbscan_top20": test["umap_hdbscan"],
                    "delta": test["umap_hdbscan"] - test["pca_ridge"],
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["version", "test_year"]).reset_index(drop=True)


def pca_ablation() -> pd.DataFrame:
    path = BASE.parent / "qwen3_prompt_return_regression_v1" / "pca_ablation_v1" / "summary" / "overall_metrics.csv"
    x = pd.read_csv(path)
    x["method_family"] = "pca_ablation_qwen"
    return x


def paired_delta(data: pd.DataFrame, family: str, metric: str = "rankic") -> pd.DataFrame:
    x = data[data["method_family"].isin(family.split(","))].copy()
    pivot = x.pivot_table(index=["model", "variant"], columns="prompt", values=metric, aggfunc="first")
    rows = []
    for prompt in PROMPTS:
        if prompt == "return" or prompt not in pivot:
            continue
        if "return" not in pivot:
            continue
        delta = pivot[prompt] - pivot["return"]
        rows.append(
            {
                "prompt": prompt,
                "prompt_zh": PROMPT_ZH[prompt],
                "n_pairs": int(delta.notna().sum()),
                f"mean_{metric}_delta_vs_return": float(delta.mean()),
                f"median_{metric}_delta_vs_return": float(delta.median()),
                "wins_vs_return": int((delta > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lh = linear_hard()
    soft = soft_cluster()
    umap = umap_hdbscan()
    pca = pca_ablation()

    # Only compare soft protocols with the same objective/model/variant.
    soft_pivot = soft.pivot_table(
        index=["model", "variant", "objective"], columns="prompt", values="rankic", aggfunc="first"
    )
    soft_rows = []
    for prompt in ["profit", "excess_return", "loss"]:
        d = soft_pivot[prompt] - soft_pivot["return"]
        soft_rows.append(
            {
                "prompt": prompt,
                "prompt_zh": PROMPT_ZH[prompt],
                "n_pairs": int(d.notna().sum()),
                "mean_rankic_delta_vs_return": float(d.mean()),
                "median_rankic_delta_vs_return": float(d.median()),
                "wins_vs_return": int((d > 0).sum()),
            }
        )
    soft_delta = pd.DataFrame(soft_rows)

    payload = {
        "source_report": str(BASE / "REPORT_ALL_RESULTS.md"),
        "protocol": {
            "years": "2018-2026, 6y train + 2y validation + 1y test",
            "return_token_baseline": "prompt=分析股票收益, same model/variant/objective",
            "leakage_control": "test-year choices frozen from validation",
            "soft_metric_note": "soft-cluster deltas use selected fold daily_rank_ic; simple_states IC is a separate platform protocol",
        },
        "method_coverage": [
            "PCA dimension ablation: none/32/64/128/256 (Qwen report experiment)",
            "PCA + MiniBatchKMeans hard cluster one-hot + Ridge",
            "PCA + distance-temperature soft assignment + cluster-return shrinkage",
            "PCA64 + UMAP8 + HDBSCAN + Ridge (loss/BGE-M3 masked-short only)",
            "No completed GMM/DBSCAN/spectral/hierarchical result is listed in the report",
        ],
        "linear_and_hard": lh.to_dict(orient="records"),
        "linear_hard_delta_vs_return": {
            "linear_ridge": {
                metric: paired_delta(lh, "linear_ridge", metric).to_dict(orient="records")
                for metric in ["rankic", "long_net", "ls_net"]
            },
            "hard_kmeans_ridge": {
                metric: paired_delta(lh, "hard_kmeans_ridge", metric).to_dict(orient="records")
                for metric in ["rankic", "long_net", "ls_net"]
            },
        },
        "soft_cluster": soft.to_dict(orient="records"),
        "soft_delta_vs_return": soft_delta.to_dict(orient="records"),
        "umap_hdbscan": umap.to_dict(orient="records"),
        "pca_ablation": pca.to_dict(orient="records"),
    }
    (OUT / "prompt_cluster_token_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Human-readable summary intentionally reports paired deltas, not a pooled
    # ranking across incompatible labels or platform protocols.
    lines = [
        "# Prompt Token、PCA 与聚类横向比较",
        "",
        f"数据源：`{BASE / 'REPORT_ALL_RESULTS.md'}`。所有新浪方向词比较使用相同 2018--2026 滚动窗口；相对收益 token 的差值在相同模型、variant、选择目标内配对。",
        "",
        "## 已覆盖方法",
        "",
        "- PCA 消融：none/32/64/128/256（报告中的 Qwen 收益 token 消融）。",
        "- 硬聚类：StandardScaler → PCA → MiniBatchKMeans（k=2/4/6/8/12）→ one-hot 簇特征 + Ridge。",
        "- 软聚类：MiniBatchKMeans 距离温度软分配 + 训练期簇收益收缩，验证期选择参数。",
        "- 密度聚类：PCA64 → UMAP8 → HDBSCAN → Ridge（报告中只对 BGE-M3 masked-short 亏损 token 完成）。",
        "- 报告未给出已完成的 GMM、DBSCAN、谱聚类或层次聚类结果，不能把它们当作已验证方法。",
        "",
        "## 相对收益 token 的 RankIC 差值",
        "",
        "| 方法 | 其他 token | 配对数 | 平均 ΔRankIC | 中位 ΔRankIC | 胜出次数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method_name, df in [("线性 Ridge", paired_delta(lh, "linear_ridge")), ("硬 KMeans + Ridge", paired_delta(lh, "hard_kmeans_ridge")), ("软聚类", soft_delta)]:
        for row in df.to_dict(orient="records"):
            lines.append(
                f"| {method_name} | {row['prompt_zh']} | {row['n_pairs']} | {row['mean_rankic_delta_vs_return']:+.5f} | {row['median_rankic_delta_vs_return']:+.5f} | {row['wins_vs_return']} |"
            )
    lines += [
        "",
        "## 线性/硬聚类的收益组合差值",
        "",
        "这里的 `long_net` 和 `ls_net` 是同一文件中的 9 年汇总收益，正值表示其他 token 高于收益 token；它们只用于同协议相对比较。",
        "",
        "| 方法 | 其他 token | Δlong_net | Δls_net |",
        "|---|---|---:|---:|",
    ]
    for prompt in ["profit", "excess_return", "loss"]:
        for family, label in [("linear_ridge", "线性 Ridge"), ("hard_kmeans_ridge", "硬 KMeans + Ridge")]:
            a = paired_delta(lh, family, "long_net").set_index("prompt").loc[prompt]
            b = paired_delta(lh, family, "ls_net").set_index("prompt").loc[prompt]
            lines.append(f"| {label} | {PROMPT_ZH[prompt]} | {a['mean_long_net_delta_vs_return']:+.4f} | {b['mean_ls_net_delta_vs_return']:+.4f} |")
    lines += [
        "",
        "## 解释",
        "",
        "收益 token 不是在所有配置、所有聚类方法中都绝对第一。方向词 span 的线性/硬聚类 RankIC 中，盈利和超额收益平均略高于收益；收益 token 的优势更多体现在部分配置的净多头/多空收益和软聚类的跨配置稳定性。亏损 token 在部分组合构造结果上很强，但不是稳定的线性 RankIC 第一。",
        "",
        "因此，横向结果支持‘收益 token 含有稳定任务相关信息’，但不能用‘所有其他 token 都显著更差’来证明。更强的证据是：在严格配对、同一滚动切分下，收益 token 在多数配置保持正 RankIC、跨年为正，并且相对无聚类基线没有依赖测试期选参。",
        "",
        "PCA/聚类的结论也应分开：PCA32 在 Qwen 线性消融中平均 RankIC 0.05352、8.67/9 年为正，明显优于原始 4096 维的 0.04425；硬 KMeans 平均增量接近零或略负；软聚类能恢复连续分层但没有超过最强 Ridge；UMAP+HDBSCAN 对亏损 token 的 Top20% 多空模拟平均仅有小幅增量，尚不足以称稳定改进。",
        "",
        "完整逐配置数据见 `prompt_cluster_token_comparison.json`。",
    ]
    (OUT / "prompt_cluster_token_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
