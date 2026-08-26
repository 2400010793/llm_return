"""Collect simple_states one-row outputs into an auditable ranking table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect_results(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*.stats.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        row = frame.iloc[-1].to_dict()
        row["result_path"] = str(path)
        rows.append(row)
    if not rows:
        raise ValueError(f"no simple_states stats found under {root}")
    result = pd.DataFrame(rows)
    for column in ("IC", "ICIR", "Ret", "Sharpe", "ICTN", "SharpeTN"):
        if column in result:
            result[f"rank_{column}"] = result[column].rank(
                method="min", ascending=False
            ).astype("Int64")
    return result.sort_values(["IC", "Sharpe"], ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    result = collect_results(args.root)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    result.to_csv(csv_path, index=False)
    payload = {
        "root": str(args.root),
        "models": int(len(result)),
        "positive_ic": int(result["IC"].gt(0).sum()),
        "positive_sharpe": int(result["Sharpe"].gt(0).sum()),
        "ranking_rule": "IC descending, then Sharpe descending; no post-hoc direction flip",
        "best_by_ic": result.iloc[0].to_dict(),
        "best_by_sharpe": result.sort_values("Sharpe", ascending=False).iloc[0].to_dict(),
        "csv": str(csv_path),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
