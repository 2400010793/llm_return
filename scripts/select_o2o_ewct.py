"""Freeze one EWCT configuration using validation-period net Sharpe only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_GAMMAS = tuple(round(value / 10, 1) for value in range(1, 10))


def _years(path: Path) -> list[int]:
    frame = pd.read_parquet(path, columns=["entry_date"])
    return sorted(
        int(value)
        for value in pd.to_datetime(frame["entry_date"], errors="coerce").dt.year.dropna().unique()
    )


def _finite(value: Any, fallback: float) -> float:
    number = float(value)
    return number if math.isfinite(number) else fallback


def selection_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    """Prefer net Sharpe, then net mean return, then lower turnover."""
    return (
        _finite(candidate["net_sharpe"], -math.inf),
        _finite(candidate["net_mean"], -math.inf),
        -_finite(candidate["mean_daily_turnover"], math.inf),
    )


def collect_candidates(manifest: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("EWCT manifest is empty")
    for row in rows:
        validation = Path(row["validation_predictions"])
        confirmation = Path(row["confirmation_predictions"])
        if _years(validation) != [2024, 2025]:
            raise ValueError(f"validation predictions are not confined to 2024-2025: {validation}")
        if _years(confirmation) != [2026]:
            raise ValueError(f"confirmation predictions are not confined to 2026: {confirmation}")
        summary_path = Path(row["validation_output_dir"]) / "summary.json"
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        for gamma in EXPECTED_GAMMAS:
            label = f"gamma_{gamma:.2f}".replace(".", "p")
            if label not in report["runs"]:
                raise ValueError(f"missing {label} in {summary_path}")
            run = report["runs"][label]
            metrics = run["metrics"]
            candidates.append({
                "task_id": int(row["task_id"]),
                "model_id": row["model_id"],
                "validation_predictions": str(validation),
                "confirmation_predictions": str(confirmation),
                "quantiles": int(row["quantiles"]),
                "gamma": gamma,
                "validation_output_dir": row["validation_output_dir"],
                "validation_summary": str(summary_path),
                "n_days": int(run["n_days"]),
                "net_sharpe": metrics["net"]["sharpe"],
                "net_mean": metrics["net"]["mean"],
                "net_annualized_return": metrics["net"]["annualized_return"],
                "gross_sharpe": metrics["gross"]["sharpe"],
                "mean_daily_turnover": metrics["execution"]["mean_daily_turnover"],
                "mean_daily_transaction_cost": metrics["execution"]["mean_daily_transaction_cost"],
            })
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = collect_candidates(args.manifest)
    ranked = sorted(candidates, key=selection_key, reverse=True)
    payload = {
        "format_version": "o2o_ewct_selection_v1",
        "selection_protocol": {
            "validation_years": [2024, 2025],
            "confirmation_years": [2026],
            "criterion": "maximize validation net Sharpe; tie-break net mean then lower turnover",
            "test_year_accessed_during_selection": False,
            "cost_model": "paper 10 bps fallback for all stocks",
            "market_return": "open_to_open_return; missing held-position returns marked zero",
            "gammas": list(EXPECTED_GAMMAS),
            "quantiles": [5, 10],
        },
        "selected": ranked[0],
        "leaderboard": ranked,
        "provenance": {
            "task_record_id": os.environ.get("TASK_RECORD_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "selected": ranked[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
