"""Build aligned prediction-level late-fusion factors.

The script never selects a weight from test outcomes.  Without explicit
validation files it emits the fixed candidate weights (including 0.5).  With
``--validation-left/right`` it selects one weight by validation daily RankIC
and applies that frozen weight to the supplied test pair.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _load(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"stock_id", "entry_date", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    result = frame[["stock_id", "entry_date", "prediction"]].copy()
    result["entry_date"] = pd.to_datetime(result["entry_date"], errors="coerce")
    result["stock_id"] = result["stock_id"].astype(str)
    result = result.rename(columns={"prediction": prefix})
    if result.duplicated(["stock_id", "entry_date"]).any():
        raise ValueError(f"{path} has duplicate stock-day predictions")
    return result.dropna(subset=[prefix, "entry_date"])


def _daily_ic(frame: pd.DataFrame, column: str) -> float:
    values = []
    for _, part in frame.groupby("entry_date", observed=True):
        if len(part) >= 5 and part[column].nunique() > 1 and part["actual"].nunique() > 1:
            value = spearmanr(part[column], part["actual"]).statistic
            if np.isfinite(value):
                values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weights", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    p.add_argument("--validation-left", type=Path)
    p.add_argument("--validation-right", type=Path)
    p.add_argument("--validation-target", default="actual")
    args = p.parse_args()
    left, right = _load(args.left, "left_prediction"), _load(args.right, "right_prediction")
    frame = left.merge(right, on=["stock_id", "entry_date"], how="inner", validate="one_to_one")
    if "actual" not in pd.read_parquet(args.left, columns=None).columns:
        raise ValueError("test predictions must include actual for audit")
    actual = pd.read_parquet(args.left, columns=["stock_id", "entry_date", "actual"])
    actual["entry_date"] = pd.to_datetime(actual["entry_date"], errors="coerce")
    actual["stock_id"] = actual["stock_id"].astype(str)
    frame = frame.merge(actual, on=["stock_id", "entry_date"], how="inner", validate="one_to_one")
    selected = None
    validation_scores = []
    if args.validation_left and args.validation_right:
        vl = _load(args.validation_left, "left_prediction")
        vr = _load(args.validation_right, "right_prediction")
        va = pd.read_parquet(args.validation_left, columns=["stock_id", "entry_date", args.validation_target]).rename(columns={args.validation_target: "actual"})
        va["entry_date"] = pd.to_datetime(va["entry_date"], errors="coerce")
        va["stock_id"] = va["stock_id"].astype(str)
        validation = vl.merge(vr, on=["stock_id", "entry_date"], how="inner", validate="one_to_one").merge(va, on=["stock_id", "entry_date"], how="inner", validate="one_to_one")
        for weight in args.weights:
            validation[f"score_{weight:.2f}"] = weight * validation.left_prediction + (1.0 - weight) * validation.right_prediction
            validation_scores.append({"weight": weight, "daily_ic": _daily_ic(validation, f"score_{weight:.2f}")})
        selected = max(validation_scores, key=lambda row: (-np.inf if not np.isfinite(row["daily_ic"]) else row["daily_ic"]))["weight"]
    outputs = []
    for weight in args.weights:
        name = f"late_fusion_{weight:.2f}"
        out = frame[["stock_id", "entry_date", "actual"]].copy()
        out["prediction"] = weight * frame.left_prediction + (1.0 - weight) * frame.right_prediction
        path = args.output_dir / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        outputs.append({"weight": weight, "path": str(path), "test_daily_ic": _daily_ic(out, "prediction"), "selected": selected == weight})
    (args.output_dir / "manifest.json").write_text(__import__("json").dumps({"selected_weight": selected, "validation": validation_scores, "outputs": outputs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"rows": len(frame), "selected_weight": selected, "outputs": len(outputs)})


if __name__ == "__main__":
    main()
