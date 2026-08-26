"""Summarize frozen Qwen linear tests without selecting on test metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_results(manifest: Path, protocol: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows, reports = [], []
    for task in read_manifest(manifest):
        output = Path(task["output"])
        if not output.is_file():
            raise ValueError(f"missing completed report: {output}")
        report = json.loads(output.read_text(encoding="utf-8"))
        reports.append({"task": task, "report": report})
        for result in report["results"]:
            rows.append({
                "protocol": protocol, "model": task["model"], "variant": task["variant"],
                "regressor": task["regressor"], "reducer": task["reducer"],
                "components": int(task["components"]), "seed": int(task["seed"]),
                "candidate_role": "huber_stability" if "huber_stability" in task["output"] else "best",
                "test_year": int(result["test_year"]),
                **{key: value for key, value in result.items()
                   if isinstance(value, (int, float)) and key != "test_year"},
            })
    return pd.DataFrame(rows), reports


def daily_ic(frame: pd.DataFrame) -> pd.DataFrame:
    values = []
    for date, group in frame.groupby("entry_date"):
        if len(group) < 5 or group["prediction"].nunique() < 2 or group["actual_return"].nunique() < 2:
            continue
        values.append({"entry_date": pd.Timestamp(date),
                       "rank_ic": group["prediction"].corr(group["actual_return"], method="spearman"),
                       "pearson_ic": group["prediction"].corr(group["actual_return"], method="pearson")})
    return pd.DataFrame(values)


def block_bootstrap_delta(left: pd.DataFrame, right: pd.DataFrame, *, samples: int = 2000,
                          block: int = 20) -> dict[str, float]:
    merged = left.merge(right, on="entry_date", suffixes=("_left", "_right"))
    delta = (merged["rank_ic_right"] - merged["rank_ic_left"]).to_numpy(float)
    if not len(delta):
        return {"mean_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "days": 0}
    rng = np.random.default_rng(42)
    starts = np.arange(len(delta))
    draws = np.empty(samples)
    blocks = int(np.ceil(len(delta) / block))
    for index in range(samples):
        chosen = rng.choice(starts, blocks, replace=True)
        positions = np.concatenate([(start + np.arange(block)) % len(delta) for start in chosen])[:len(delta)]
        draws[index] = np.mean(delta[positions])
    return {"mean_delta": float(np.mean(delta)), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "days": int(len(delta))}


def prediction_entry(item: dict[str, object]) -> tuple[dict[str, str], Path]:
    task, report = item["task"], item["report"]
    return task, Path(str(report["stock_day_predictions"]))


def enrich_raw(source: Path, panel_path: Path, output: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(source)
    panel = pd.read_parquet(panel_path, columns=["stock_id", "entry_date", "next_day_open_to_open_return"])
    panel["entry_date"] = pd.to_datetime(panel["entry_date"])
    raw = panel.groupby(["stock_id", "entry_date"], as_index=False)["next_day_open_to_open_return"].first()
    result = predictions.drop(columns=["actual_return"], errors="ignore").merge(
        raw, on=["stock_id", "entry_date"], how="left", validate="one_to_one"
    ).rename(columns={"next_day_open_to_open_return": "actual_return"})
    if result["actual_return"].isna().any():
        raise ValueError("raw O2O return enrichment produced missing values")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path, required=True)
    parser.add_argument("--rolling-manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cluster-strict", type=Path, required=True)
    parser.add_argument("--cluster-rolling", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    strict, strict_reports = read_results(args.strict_manifest, "strict_6_2_1")
    rolling, rolling_reports = read_results(args.rolling_manifest, "rolling_3_1_1")
    args.output_root.mkdir(parents=True, exist_ok=True)
    strict.to_csv(args.output_root / "strict_2026_metrics.csv", index=False)
    rolling.to_csv(args.output_root / "rolling_3_1_1_yearly_metrics.csv", index=False)

    selected = json.loads(args.selected.read_text(encoding="utf-8"))["selected"]
    best_qwen = max((row for row in selected if row["model"] == "qwen3_embedding_8b"),
                    key=lambda row: (float(row["rank_ic_mean"]), float(row["oos_r2_vs_historical_mean"]), -float(row["mae"])))
    def match(items: list[dict[str, object]], candidate: dict[str, object]):
        return next(item for item in items if "huber_stability" not in item["task"]["output"]
                    and all(str(item["task"][key]) == str(candidate[key])
                            for key in ("model", "variant", "regressor", "reducer", "components")))
    strict_best = match(strict_reports, best_qwen)
    rolling_best = match(rolling_reports, best_qwen)
    _, strict_path = prediction_entry(strict_best)
    _, rolling_path = prediction_entry(rolling_best)
    strict_raw = args.output_root / "strict_2026_stock_day_predictions.parquet"
    rolling_raw = args.output_root / "stock_day_predictions.parquet"
    enrich_raw(strict_path, args.panel, strict_raw)
    rolling_frame = enrich_raw(rolling_path, args.panel, rolling_raw)

    baseline_rows = []
    for protocol, item in (("strict_6_2_1", strict_best), ("rolling_3_1_1", rolling_best)):
        _, path = prediction_entry(item)
        predictions = pd.read_parquet(path)
        means = {int(row["test_year"]): float(row["historical_mean_benchmark"])
                 for row in item["report"]["results"]}
        for year, group in predictions.groupby("test_year"):
            actual = group["actual_return"].to_numpy(float)
            for name, value in (("zero", 0.0), ("historical_mean", means[int(year)])):
                forecast = np.full(len(actual), value)
                mse = float(np.mean(np.square(actual - forecast)))
                denominator = float(np.sum(np.square(actual - np.mean(actual))))
                baseline_rows.append({
                    "protocol": protocol, "test_year": int(year), "baseline": name,
                    "mse": mse, "mae": float(np.mean(np.abs(actual - forecast))),
                    "oos_r2_vs_test_mean": 1.0 - float(np.sum(np.square(actual - forecast))) / denominator
                    if denominator > 0 else np.nan,
                    "direction_accuracy": float(np.mean((forecast > 0) == (actual > 0))),
                    "rank_ic_mean": np.nan,
                })
    pd.DataFrame(baseline_rows).to_csv(args.output_root / "constant_baseline_metrics.csv", index=False)

    monthly = daily_ic(pd.read_parquet(rolling_path))
    monthly["month"] = monthly["entry_date"].dt.to_period("M").astype(str)
    monthly.groupby("month", as_index=False).agg(
        rank_ic=("rank_ic", "mean"), pearson_ic=("pearson_ic", "mean"),
        days=("rank_ic", "size"), positive_rank_ic=("rank_ic", lambda x: float((x > 0).mean())),
    ).to_csv(args.output_root / "monthly_rankic.csv", index=False)

    best_strict = [item for item in strict_reports if "huber_stability" not in item["task"]["output"]]
    qwen_daily = daily_ic(pd.read_parquet(strict_path))
    zero_daily = qwen_daily[["entry_date"]].copy(); zero_daily["rank_ic"] = 0.0
    absolute = block_bootstrap_delta(zero_daily, qwen_daily)
    paired = [{"left_model": "zero_ic", "right_model": "qwen3_embedding_8b",
               "right_variant": best_qwen["variant"], **absolute}]
    for item in best_strict:
        task, path = prediction_entry(item)
        if task["model"] == "qwen3_embedding_8b":
            continue
        result = block_bootstrap_delta(daily_ic(pd.read_parquet(path)), qwen_daily)
        paired.append({"left_model": task["model"], "right_model": "qwen3_embedding_8b",
                       "right_variant": best_qwen["variant"], **result})
    pd.DataFrame(paired).to_csv(args.output_root / "paired_model_comparison.csv", index=False)

    cluster_frames = []
    for protocol, root in (("strict_6_2_1", args.cluster_strict), ("rolling_3_1_1", args.cluster_rolling)):
        frame = pd.read_csv(root / "cluster_linear_increment.csv")
        frame.insert(0, "protocol", protocol); cluster_frames.append(frame)
    pd.concat(cluster_frames, ignore_index=True).to_csv(args.output_root / "cluster_linear_increment.csv", index=False)
    diagnostics = []
    for protocol, root in (("strict_6_2_1", args.cluster_strict), ("rolling_3_1_1", args.cluster_rolling)):
        frame = pd.read_csv(root / "cluster_diagnostics.csv")
        frame.insert(0, "protocol", protocol); diagnostics.append(frame)
    pd.concat(diagnostics, ignore_index=True).to_csv(args.output_root / "cluster_diagnostics.csv", index=False)

    stable = rolling[(rolling["model"] == "qwen3_embedding_8b") &
                     (rolling["candidate_role"] == "huber_stability")]
    stability = stable.groupby(["variant", "seed"], as_index=False).agg(
        mean_rank_ic=("rank_ic_mean", "mean"), positive_years=("rank_ic_mean", lambda x: int((x > 0).sum())),
        years=("test_year", "nunique"),
    )
    stability.to_csv(args.output_root / "huber_seed_stability.csv", index=False)
    report = [
        "# Qwen 2018--2026 线性收益预测报告", "",
        f"- 冻结最佳Qwen表示：`{best_qwen['variant']}`。",
        "- 主检验为6/2/1的2026完全留出测试；3/1/1仅用于跨年稳健性。",
        "- 监督模型仅包含Ridge和线性Huber SGD。", "",
        "## 当前状态", "",
        f"- 严格测试配置：{len(strict)}行。",
        f"- 辅助滚动配置：{len(rolling)}行。",
        f"- 最佳Qwen滚动股票日预测：{len(rolling_frame)}行。", "",
        "详细数值见同目录CSV；结论必须按既定强预测能力标准逐项核验。",
    ]
    (args.output_root / "report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"best_qwen": best_qwen, "strict_rows": len(strict),
                      "rolling_rows": len(rolling)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
