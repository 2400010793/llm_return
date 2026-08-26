"""Summarize the completed Qwen Sina Ridge PCA ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def weighted(frame: pd.DataFrame, value: str, weight: str) -> float:
    keep = frame[value].notna() & frame[weight].gt(0)
    return float(np.average(frame.loc[keep, value], weights=frame.loc[keep, weight]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    overall, yearly = [], []
    for row in manifest.itertuples(index=False):
        report = json.loads(Path(row.output).read_text(encoding="utf-8"))
        folds = pd.DataFrame(report["results"])
        reducer = "none" if row.reducer == "none" else f"pca{int(row.components)}"
        for fold in report["results"]:
            yearly.append({
                "variant": row.variant, "representation": row.representation,
                "reducer": reducer, "test_year": int(fold["test_year"]),
                "rank_ic": fold["rank_ic_mean"],
                "direction_accuracy": fold["direction_accuracy"],
                "oos_r2": fold["oos_r2_vs_historical_mean"],
                "n": fold["n"], "rank_ic_days": fold["rank_ic_days"],
            })
        overall.append({
            "variant": row.variant, "representation": row.representation,
            "reducer": reducer,
            "rank_ic": weighted(folds, "rank_ic_mean", "rank_ic_days"),
            "positive_years": int(folds["rank_ic_mean"].gt(0).sum()),
            "direction_accuracy": weighted(folds, "direction_accuracy", "n"),
            "oos_r2": weighted(folds, "oos_r2_vs_historical_mean", "n"),
            "test_stock_days": int(folds["n"].sum()),
        })
    overall_frame = pd.DataFrame(overall)
    baseline = overall_frame.loc[overall_frame["reducer"].eq("none"), [
        "variant", "representation", "rank_ic", "direction_accuracy", "oos_r2",
    ]].rename(columns={
        "rank_ic": "none_rank_ic", "direction_accuracy": "none_direction_accuracy",
        "oos_r2": "none_oos_r2",
    })
    comparison = overall_frame.merge(
        baseline, on=["variant", "representation"], how="left", validate="many_to_one",
    )
    comparison["rank_ic_delta_vs_none"] = comparison["rank_ic"] - comparison["none_rank_ic"]
    comparison["accuracy_delta_vs_none"] = comparison["direction_accuracy"] - comparison["none_direction_accuracy"]
    comparison["oos_r2_delta_vs_none"] = comparison["oos_r2"] - comparison["none_oos_r2"]
    comparison = comparison.sort_values(["variant", "representation", "rank_ic"], ascending=[True, True, False])

    args.output_root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_root / "overall_metrics.csv", index=False)
    pd.DataFrame(yearly).to_csv(args.output_root / "yearly_metrics.csv", index=False)
    winners = comparison.sort_values("rank_ic", ascending=False).groupby(
        ["variant", "representation"], as_index=False
    ).first()
    winners.to_csv(args.output_root / "best_reducer_by_representation.csv", index=False)
    audit = {
        "configs": len(comparison), "expected_configs": 30,
        "all_complete": len(comparison) == 30,
        "pca_rankic_wins": int(
            winners["rank_ic_delta_vs_none"].gt(0).sum()
        ),
        "representations": len(winners),
    }
    (args.output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
