"""Average aligned stock-day regression predictions across repeated runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["stock_id", "entry_date", "test_year"]


def ensemble_predictions(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    if len(paths) < 2:
        raise ValueError("an ensemble needs at least two prediction files")
    aligned: pd.DataFrame | None = None
    for index, path in enumerate(paths):
        frame = pd.read_parquet(path)
        required = {*KEYS, "actual_return", "prediction"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        values = frame[[*KEYS, "actual_return", "prediction"]].copy()
        values["entry_date"] = pd.to_datetime(values["entry_date"], errors="coerce")
        if values[KEYS].isna().any().any() or values.duplicated(KEYS).any():
            raise ValueError(f"{path} has invalid or duplicate stock-day keys")
        values = values.rename(columns={
            "actual_return": f"actual_return_{index}",
            "prediction": f"prediction_{index}",
        })
        aligned = values if aligned is None else aligned.merge(
            values, on=KEYS, how="inner", validate="one_to_one"
        )
    assert aligned is not None
    expected_rows = len(pd.read_parquet(paths[0], columns=KEYS))
    if len(aligned) != expected_rows:
        raise ValueError(
            f"prediction key mismatch: aligned {len(aligned)}/{expected_rows} rows"
        )
    actual_columns = [f"actual_return_{i}" for i in range(len(paths))]
    prediction_columns = [f"prediction_{i}" for i in range(len(paths))]
    actual = aligned[actual_columns].to_numpy(dtype=float)
    reference = actual[:, [0]]
    if not np.allclose(actual, reference, rtol=0.0, atol=1e-12, equal_nan=True):
        raise ValueError("actual returns differ across seed prediction files")
    predictions = aligned[prediction_columns].to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("seed predictions must all be finite")
    output = aligned[KEYS].copy()
    output["actual_return"] = actual[:, 0]
    output["prediction"] = predictions.mean(axis=1)
    output["prediction_seed_std"] = predictions.std(axis=1, ddof=0)
    output["ensemble_size"] = len(paths)
    summary = {
        "inputs": [str(path) for path in paths],
        "rows": int(len(output)),
        "ensemble_size": len(paths),
        "prediction_mean": float(output["prediction"].mean()),
        "prediction_std": float(output["prediction"].std(ddof=0)),
        "mean_seed_disagreement": float(output["prediction_seed_std"].mean()),
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, summary = ensemble_predictions(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    report = {**summary, "output": str(args.output)}
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
