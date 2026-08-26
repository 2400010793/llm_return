"""Select the full two-stage configuration on 2024 and confirm it on 2025."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def score(row: dict[str, Any]) -> tuple[float, float, float]:
    metrics = row["validation_metrics"]
    rank_ic = float(metrics.get("rank_ic_mean", float("nan")))
    r2 = float(metrics.get("oos_r2_vs_historical_mean", float("nan")))
    mse = float(metrics.get("mse", float("nan")))
    return (
        rank_ic if np.isfinite(rank_ic) else -np.inf,
        r2 if np.isfinite(r2) else -np.inf,
        -mse if np.isfinite(mse) else -np.inf,
    )


def flatten(report: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    representation = report["representation"]
    window = "tune2024" if representation["windows"]["predictor_validation"] == [2024] else "confirm2025"
    rows = []
    for candidate in report["candidates"]:
        rows.append({
            **candidate,
            "window": window,
            "model": representation["model"],
            "variant": representation["variant"],
            "gate": representation["gate"],
            "source": str(source),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "window", "model", "variant", "gate_mode",
        "representation_size", "gate_hidden_size", "reducer", "components",
        "regressor", "params", "rank_ic_mean", "rank_ic_t_stat",
        "oos_r2_vs_historical_mean", "mse", "config_key", "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            metrics = row["validation_metrics"]
            writer.writerow({
                "rank": rank, "window": row["window"], "model": row["model"],
                "variant": row["variant"], "gate_mode": row["gate"]["mode"],
                "representation_size": row["gate"]["representation_size"],
                "gate_hidden_size": row["gate"]["gate_hidden_size"],
                "reducer": row["reducer"], "components": row["components"],
                "regressor": row["regressor"],
                "params": json.dumps(row["params"], ensure_ascii=False, sort_keys=True),
                "rank_ic_mean": metrics.get("rank_ic_mean"),
                "rank_ic_t_stat": metrics.get("rank_ic_t_stat"),
                "oos_r2_vs_historical_mean": metrics.get("oos_r2_vs_historical_mean"),
                "mse": metrics.get("mse"), "config_key": row["config_key"],
                "source": row["source"],
            })


def main() -> None:
    manifest = Path("configs/generated/two_stage_gate_v1_predictors.tsv")
    with manifest.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    reports = [Path(row["predictor_output"]) for row in manifest_rows]
    missing = [str(path) for path in reports if not path.is_file()]
    if missing:
        raise ValueError(f"predictor reports are incomplete ({len(missing)} missing): {missing[:8]}")
    rows = []
    for path in reports:
        rows.extend(flatten(json.loads(path.read_text(encoding="utf-8")), path))
    tune = sorted((row for row in rows if row["window"] == "tune2024"), key=score, reverse=True)
    confirm_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["window"] == "confirm2025":
            confirm_by_key.setdefault(row["config_key"], []).append(row)
    if not tune:
        raise ValueError("no 2024 selection candidates found")
    selected = tune[0]
    matches = confirm_by_key.get(selected["config_key"], [])
    if len(matches) != 1:
        raise ValueError(f"selected config has {len(matches)} exact 2025 matches")
    confirmed = matches[0]
    output = {
        "format_version": "two_stage_gate_selection_confirmation_v1",
        "protocol": {
            "selection": "maximum stock-day Rank IC on 2024",
            "confirmation": "same complete configuration key evaluated on 2025",
            "test_year_accessed": False,
        },
        "counts": {
            "reports": len(reports), "all_candidates": len(rows),
            "tune_candidates": len(tune),
        },
        "selected_on_2024": selected,
        "confirmed_on_2025": confirmed,
    }
    Path("reports/two_stage_gate_v1_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    write_csv(Path("reports/two_stage_gate_v1_2024_ranking.csv"), tune)
    write_csv(Path("reports/two_stage_gate_v1_selected_2025.csv"), [confirmed])
    print(json.dumps({
        "selected_2024_rank_ic": score(selected)[0],
        "confirmed_2025_rank_ic": score(confirmed)[0],
        "config_key": selected["config_key"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
