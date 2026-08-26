"""Summarize completed yearly stock-graph classification reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*_test*_seed*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        result = report["result"]
        rows.append({
            "path": str(path),
            "mode": result["mode"],
            "test_year": result["test_year"],
            "n": result["n_test_announcements"],
            "stock_days": result["n_test_nodes"],
            "accuracy": result["accuracy"],
            "majority_accuracy": result["majority_accuracy"],
            "stock_day_accuracy": result["stock_day_metrics"]["accuracy"],
            "stock_day_majority_accuracy": result["stock_day_metrics"]["majority_accuracy"],
            "selected_epoch": result["selected_epoch"],
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no graph reports found under {args.input_dir}")
    summaries = []
    for mode, group in frame.groupby("mode", sort=True):
        summaries.append({
            "mode": mode,
            "years": int(len(group)),
            "announcement_rows": int(group["n"].sum()),
            "stock_days": int(group["stock_days"].sum()),
            "weighted_accuracy": float((group["accuracy"] * group["n"]).sum() / group["n"].sum()),
            "weighted_majority_accuracy": float(
                (group["majority_accuracy"] * group["n"]).sum() / group["n"].sum()
            ),
            "stock_day_weighted_accuracy": float(
                (group["stock_day_accuracy"] * group["stock_days"]).sum()
                / group["stock_days"].sum()
            ),
            "above_yearly_majority": int((group["accuracy"] > group["majority_accuracy"]).sum()),
        })
    output = {
        "format_version": "stock_graph_classification_summary_v1",
        "reports": int(len(frame)),
        "expected_reports": 27,
        "complete": len(frame) == 27,
        "summary": summaries,
        "yearly": frame.sort_values(["mode", "test_year"]).to_dict("records"),
        "caveat": (
            "Industry labels are current rather than historical point-in-time; "
            "industry results are exploratory. Self and randomized controls share the same sample."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
