"""Audit the frozen dynamic-prompt replication inputs and current outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import FEATURE_GROUPS
from src.evaluation.artifacts import atomic_json, file_fingerprint


def manifest_status(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    completed = [row for row in rows if Path(row["output"]).is_file()]
    return {
        "manifest": str(path),
        "tasks": len(rows),
        "completed": len(completed),
        "missing_task_ids": [int(row["task_id"]) for row in rows if not Path(row["output"]).is_file()],
    }


def segment_gate_audit(path: Path) -> list[dict[str, object]]:
    group_order = list(FEATURE_GROUPS["title_body_full_concat"])
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    for task in manifest:
        report_path = Path(task["output"])
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = report["results"][0]
        gate = result["token_gate"]
        record: dict[str, object] = {
            "model": task["model"],
            "variant": task["variant"],
            "gate_method": task["gate_method"],
            "group_order": group_order,
            "validation_auc": result["validation_metrics"]["auc"],
        }
        for scope in ("fit", "all_train"):
            positions = gate[scope]["selected_positions_zero_based"]
            record[f"{scope}_selected_positions_zero_based"] = positions
            record[f"{scope}_selected_groups"] = [group_order[position] for position in positions]
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path,
        default=Path("data/processed/cninfo_full_classification_panel.parquet"),
    )
    parser.add_argument(
        "--prompt-root", type=Path,
        default=Path("data/processed/prompt_token_embeddings_v5"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/replication_start_audit.json"),
    )
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    dates = pd.to_datetime(panel["entry_date"], errors="coerce")
    target = pd.to_numeric(panel["next_day_return"], errors="coerce")
    row_index = pd.to_numeric(panel["row_index"], errors="raise").to_numpy(dtype=np.int64)
    if len(np.unique(row_index)) != len(row_index):
        raise ValueError("panel row_index is not unique")

    prompt_directories = sorted(args.prompt_root.glob("shard-*/roberta/masked_short"))
    prompt_rows = 0
    prompt_row_indexes: list[np.ndarray] = []
    sequence_truncated = 0
    for directory in prompt_directories:
        metadata = pd.read_json(directory / "metadata.jsonl", lines=True)
        matrix = np.load(directory / "prompt_token_embeddings.npy", mmap_mode="r")
        if matrix.shape != (len(metadata), 28, 768):
            raise ValueError(f"unexpected prompt shape in {directory}: {matrix.shape}")
        local_rows = pd.to_numeric(metadata["row_index"], errors="raise").to_numpy(dtype=np.int64)
        prompt_rows += len(local_rows)
        prompt_row_indexes.append(local_rows)
        sequence_truncated += int(metadata["sequence_truncated"].astype(bool).sum())
    all_prompt_rows = np.concatenate(prompt_row_indexes) if prompt_row_indexes else np.array([], dtype=np.int64)

    year_counts = dates.dt.year.value_counts().sort_index()
    report = {
        "format_version": "replication_start_audit_v1",
        "scope": "CNINFO announcement method replication; not strict news-data replication",
        "panel": {
            "fingerprint": file_fingerprint(args.panel, hash_content=True),
            "rows": len(panel),
            "stocks": int(panel["stock_id"].nunique()),
            "entry_dates": int(dates.nunique()),
            "year_counts": {str(int(year)): int(count) for year, count in year_counts.items()},
            "last_entry_date": str(dates.max().date()),
            "unique_row_index": bool(len(np.unique(row_index)) == len(row_index)),
            "finite_next_day_targets": int(np.isfinite(target.to_numpy(dtype=float)).sum()),
            "stock_day_rows": int(panel.assign(_date=dates).drop_duplicates(["stock_id", "_date"]).shape[0]),
        },
        "prompt_embeddings": {
            "model": "roberta", "variant": "masked_short",
            "shards": len(prompt_directories), "rows": prompt_rows,
            "unique_rows": int(len(np.unique(all_prompt_rows))),
            "covers_panel_rows": bool(np.array_equal(np.sort(all_prompt_rows), np.sort(row_index))),
            "shape_per_row": [28, 768],
            "sequence_truncated_rows": sequence_truncated,
            "sequence_truncated_rate": sequence_truncated / prompt_rows if prompt_rows else None,
        },
        "segment_gates": segment_gate_audit(
            Path("configs/generated/pooled_next1_gating_seed42.tsv")
        ),
        "regression_screening": [
            manifest_status(Path("configs/generated/pooled_regression_screen_roberta_masked.tsv")),
            manifest_status(Path("configs/generated/pooled_regression_screen_bge_m3_masked.tsv")),
        ],
        "warnings": [
            "2026 is incomplete and has already been inspected; treat it as exploratory.",
            "CNINFO announcements are not equivalent to the original paper's financial news.",
            "Derive selected segment names from persisted indices and feature group order.",
        ],
    }
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
