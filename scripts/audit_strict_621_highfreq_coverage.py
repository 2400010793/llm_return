#!/usr/bin/env python3
"""Audit whether the shared metrics can support a strict 6+2+1 calendar split."""
from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq


SHARE = Path("/data/alpha_team2/shares/260825")
OUT = Path("/mnt/lustre3/home/gaozh/llm_return/reports/intraday_token_correspondence")


def years(path: Path) -> dict:
    table = pq.ParquetFile(path)
    # The shared wide tables store datetime as the pandas index column.
    frame = pd.read_parquet(path, columns=[])
    idx = pd.to_datetime(frame.index, errors="coerce")
    values = sorted({int(x) for x in idx.year if pd.notna(x)})
    return {"min": str(idx.min()), "max": str(idx.max()), "years": values, "rows": int(len(idx))}


def main() -> None:
    metrics = {
        "intraday_realized_volatility": years(SHARE / "realized_volatility_cumulative.parquet"),
        "intraday_spread_bps": years(SHARE / "bid_ask_spread_bps.parquet"),
        "pe": years(SHARE / "DZ_DInd_pe.parquet"),
        "pb": years(SHARE / "DZ_DInd_pb.parquet"),
        "ps": years(SHARE / "DZ_DInd_ps.parquet"),
        "evtoebitda": years(SHARE / "DZ_DInd_evtoebitda.parquet"),
    }
    split = {
        "test_year": 2026,
        "fit_years": list(range(2018, 2024)),
        "validation_years": [2024, 2025],
        "test_years": [2026],
        "intraday_effective_fit_years": [year for year in range(2018, 2024) if year in metrics["intraday_realized_volatility"]["years"]],
        "intraday_effective_protocol": "3+2+1 after dropping pre-2021 rows; not a full 6+2+1",
        "valuation_effective_protocol": "full 6+2+1 calendar coverage",
        "earliest_full_intraday_test_year": 2029,
        "earliest_full_intraday_test_year_reason": "requires six historical calendar years before the 2021 data start",
    }
    payload = {"metrics": metrics, "split": split}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "strict_621_highfreq_coverage.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 高频数据 6+2+1 覆盖审计",
        "",
        "## 2026 严格日历窗口",
        "",
        "- 训练：2018--2023",
        "- 验证：2024--2025",
        "- 测试：2026",
        "",
        "## 数据覆盖",
        "",
        "| 数据 | 起始 | 结束 | 可用年份 |",
        "|---|---|---|---|",
    ]
    labels = {
        "intraday_realized_volatility": "累计实现波动率",
        "intraday_spread_bps": "买卖价差 bps",
        "pe": "PE",
        "pb": "PB",
        "ps": "PS",
        "evtoebitda": "EV/EBITDA",
    }
    for key, value in metrics.items():
        lines.append(f"| {labels[key]} | {value['min']} | {value['max']} | {','.join(map(str, value['years']))} |")
    lines += [
        "",
        "## 结论",
        "",
        "分钟波动率和价差从 2021-03-10 开始。2026 的日历 6+2+1 窗口虽然仍可建立，但删除没有高频标签的旧新闻后，训练年实际只有 2021--2023 三年，因此只能称为有效样本 3+2+1，不能称完整 6+2+1。",
        "",
        "如果必须保持严格 6+2+1，分钟数据需要补齐 2018--2020；在当前共享文件下，最早可以拥有完整六年高频训练历史的测试年份是 2029。估值数据覆盖 2010--2026，因此估值任务可以在 2026 使用完整 6+2+1。",
        "",
        "允许砍掉旧新闻时，应在结果中同时报告日历窗口和有效样本年份，不能只报告一个 6+2+1 标签。",
    ]
    (OUT / "strict_621_highfreq_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
