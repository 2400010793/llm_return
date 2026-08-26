"""Apply one validation-frozen EWCT configuration to the 2026 confirmation sample."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--market-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-report", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    protocol = selection["selection_protocol"]
    if protocol.get("test_year_accessed_during_selection") is not False:
        raise ValueError("selection is not marked as confirmation-blind")
    chosen = selection["selected"]
    predictions = Path(chosen["confirmation_predictions"])
    frame = pd.read_parquet(predictions, columns=["entry_date"])
    years = sorted(
        int(value)
        for value in pd.to_datetime(frame["entry_date"], errors="coerce").dt.year.dropna().unique()
    )
    if years != [2026]:
        raise ValueError(f"confirmation predictions are not confined to 2026: {years}")

    label = (
        f"{chosen['model_id']}_q{chosen['quantiles']}_gamma{chosen['gamma']:.1f}"
        .replace(".", "p")
    )
    output_dir = args.output_root / label
    command = [
        sys.executable, "scripts/run_portfolio_strategy.py", str(predictions),
        "--market-data", str(args.market_data),
        "--market-return-column", "open_to_open_return",
        "--output-dir", str(output_dir),
        "--quantiles", str(chosen["quantiles"]),
        "--min-stocks-per-day", str(chosen["quantiles"]),
        "--gammas", str(chosen["gamma"]),
        "--can-buy-column", "can_buy", "--can-sell-column", "can_sell",
        "--eligible-column", "eligible_signal", "--missing-return", "zero",
        "--cost-model", "paper",
    ]
    subprocess.run(command, check=True)
    strategy_summary = output_dir / "summary.json"
    confirmation = {
        "format_version": "o2o_ewct_confirmation_v1",
        "selection": str(args.selection),
        "selected_validation_configuration": chosen,
        "confirmation_years": years,
        "confirmation_strategy_summary": str(strategy_summary),
        "confirmation_metrics": json.loads(strategy_summary.read_text(encoding="utf-8")),
        "provenance": {
            "task_record_id": os.environ.get("TASK_RECORD_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    args.confirmation_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.confirmation_report.with_suffix(args.confirmation_report.suffix + ".tmp")
    temporary.write_text(json.dumps(confirmation, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.confirmation_report)
    print(json.dumps({"output": str(args.confirmation_report), "strategy": str(strategy_summary)}))


if __name__ == "__main__":
    main()
