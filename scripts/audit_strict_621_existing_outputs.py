#!/usr/bin/env python3
"""Audit the completed Token/Body outputs against the strict 6+2+1 design."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path("/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/aligned_masked_short_v1")
OUT = Path("reports/intraday_token_correspondence")
PANEL = Path("data/processed/prompt_factor_labels/prompt_factor_label_panel_v2.parquet")


def collect(rep: str) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    bad: list[dict] = []
    for path in sorted((BASE / rep).glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        design = payload.get("design", {})
        result = payload.get("results", [{}])[0]
        record = {
            "representation": rep,
            "path": str(path),
            "target": design.get("target"),
            "fit_years": design.get("fit_years"),
            "validation_years": design.get("validation_years"),
            "test_years": design.get("test_years"),
            "test_year": result.get("test_year"),
            "n_fit_announcements": result.get("n_fit_announcements"),
            "n_validation_announcements": result.get("n_validation_announcements"),
            "n_test_announcements": result.get("n_test_announcements"),
            "rank_ic": result.get("rank_ic_mean"),
            "rank_ic_ir": result.get("rank_ic_information_ratio"),
        }
        rows.append(record)
        if (
            design.get("fit_years") != 6
            or design.get("validation_years") != 2
            or design.get("test_years") != 1
            or result.get("fit_years") != list(range(2018, 2024))
            or result.get("validation_years") != [2024, 2025]
            or result.get("test_year") != 2026
        ):
            bad.append(record)
    return rows, bad


def main() -> None:
    token, token_bad = collect("regression_token_v2")
    body, body_bad = collect("regression_body_v2")
    panel = pd.read_parquet(PANEL, columns=["entry_date"])
    years = pd.to_datetime(panel["entry_date"], errors="coerce").dt.year
    panel_counts = years.value_counts().sort_index().to_dict()
    payload = {
        "protocol": {
            "calendar_fit_years": list(range(2018, 2024)),
            "calendar_validation_years": [2024, 2025],
            "calendar_test_years": [2026],
            "fit_year_count": 6,
            "validation_year_count": 2,
            "test_year_count": 1,
        },
        "panel_rows_by_year": {str(k): int(v) for k, v in panel_counts.items()},
        "outputs": {
            "token": {"tasks": len(token), "protocol_violations": len(token_bad)},
            "body": {"tasks": len(body), "protocol_violations": len(body_bad)},
        },
        "token_records": token,
        "body_records": body,
        "violations": token_bad + body_bad,
        "high_frequency_policy": {
            "metric_start": "2021-03-10",
            "effective_fit_years_after_dropping_old_news": [2021, 2022, 2023],
            "effective_protocol": "3+2+1",
            "full_6_plus_2_plus_1_requires_test_year": 2029,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "strict_621_existing_outputs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 已有 Token/Body 结果严格 6+2+1 审计",
        "",
        "## 协议",
        "",
        "- 训练：2018--2023（6 年）",
        "- 验证：2024--2025（2 年）",
        "- 测试：2026（1 年）",
        "",
        "## 已有输出",
        "",
        f"- Token：{len(token)} 个任务，协议违规 {len(token_bad)} 个。",
        f"- Body：{len(body)} 个任务，协议违规 {len(body_bad)} 个。",
        "- 两套结果的 `result.json` 均为 2026 测试折，模型选择只使用 2024--2025 验证窗口，最终训练为 2018--2025 后测试 2026。",
        "",
        "## 关键区别",
        "",
        "已有收益率、日频波动率、事件冲击等回归结果可以继续称为完整 6+2+1。这里的平均 RankIC 是模型预测值与目标标签的每日横截面 Spearman IC；聚类报告中的目标是 `forward_compounded_return_3d`，即未来 3 个交易日复合收益率。",
        "",
        "分钟波动率和买卖价差文件从 2021-03-10 才开始。为构造高频标签而删除此前新闻后，2018--2023 训练窗口只剩 2021--2023，因此应标为有效 3+2+1，不能伪称完整 6+2+1。估值文件从 2010 年开始，不需要删旧新闻。",
        "",
        "## 执行口径",
        "",
        "高频任务按 `entry_date` 过滤到指标可覆盖日期，并要求目标值非空；验证和测试窗口仍固定为 2024--2025 与 2026。所有 Token/Body 采用同一行过滤、同一标签、同一滚动窗口，避免样本选择造成表示比较偏差。",
        "",
        "详细覆盖范围见 `strict_621_highfreq_coverage.md`。",
    ]
    (OUT / "strict_621_existing_outputs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"token_tasks": len(token), "body_tasks": len(body), "violations": len(token_bad) + len(body_bad)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
