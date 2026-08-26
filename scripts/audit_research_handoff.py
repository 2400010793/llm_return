#!/usr/bin/env python3
"""Emit a read-only, machine-readable snapshot of the active research workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(
    "/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026"
)
DEFAULT_CNINFO_PANEL = Path(
    "/mnt/lustre3/home/gaozh/llm_return/data/processed/"
    "cninfo_full_classification_panel_2010_2026.parquet"
)
DEFAULT_REPORT = DEFAULT_DATA_ROOT / (
    "single_stock_cninfo_v1/prompt_positive_v1/REPORT_ALL_RESULTS.md"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parquet_shape(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    return {
        "path": str(path),
        "exists": True,
        "rows": metadata.num_rows,
        "columns": metadata.num_columns,
        "bytes": path.stat().st_size,
    }


def embedding_family(root: Path, dataset: str, model: str) -> dict[str, Any]:
    directory = root / dataset / model
    completed = sorted(
        int(marker.parent.name.removeprefix("shard-"))
        for marker in directory.glob("shard-*/COMPLETED")
        if marker.parent.name.removeprefix("shard-").isdigit()
    )
    expected = 256
    completed_set = set(completed)
    return {
        "path": str(directory),
        "expected_shards": expected,
        "completed_shards": len(completed),
        "missing_shards": [index for index in range(expected) if index not in completed_set],
        "complete": len(completed) == expected,
    }


def report_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    with path.open(encoding="utf-8") as handle:
        lines = sum(1 for _ in handle)
    return {
        "path": str(path),
        "exists": True,
        "lines": lines,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def slurm_status(user: str) -> dict[str, Any]:
    command = [
        "squeue", "-u", user, "-h", "-o",
        "%T|%A|%F|%i|%j|%C|%m|%l|%r",
    ]
    try:
        completed = subprocess.run(
            command, check=True, text=True, capture_output=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return {"available": False, "error": str(error)}
    jobs = []
    totals: dict[str, dict[str, int]] = {}
    for line in completed.stdout.splitlines():
        if not line:
            continue
        state, array_id, parent_id, job_id, name, cpus, memory, limit, reason = line.split("|", 8)
        jobs.append({
            "state": state,
            "array_id": array_id,
            "parent_id": parent_id,
            "job_id": job_id,
            "name": name,
            "cpus_per_visible_entry": int(cpus),
            "memory": memory,
            "time_limit": limit,
            "reason": reason,
        })
        summary = totals.setdefault(state, {"visible_entries": 0, "cpus": 0})
        summary["visible_entries"] += 1
        summary["cpus"] += int(cpus)
    return {"available": True, "user": user, "totals": totals, "jobs": jobs}


def build_status(
    *, data_root: Path, cninfo_panel: Path, report: Path,
    include_slurm: bool, slurm_user: str,
) -> dict[str, Any]:
    neutral_root = data_root / "neutral_masked_short_v1"
    embedding_root = neutral_root / "embeddings"
    sina_panel = data_root / "classification/sina_full_classification_panel.parquet"
    inputs = {
        dataset: read_json(neutral_root / "inputs" / dataset / "manifest.json")
        for dataset in ("sina", "cninfo")
    }
    status: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "read_only": True,
        "panels": {
            "sina": parquet_shape(sina_panel),
            "cninfo": parquet_shape(cninfo_panel),
        },
        "neutral_masked_short_inputs": inputs,
        "neutral_masked_short_embeddings": {
            dataset: {
                model: embedding_family(embedding_root, dataset, model)
                for model in ("roberta", "bge_m3")
            }
            for dataset in ("sina", "cninfo")
        },
        "master_report": report_status(report),
    }
    if include_slurm:
        status["slurm"] = slurm_status(slurm_user)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cninfo-panel", type=Path, default=DEFAULT_CNINFO_PANEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--include-slurm", action="store_true")
    parser.add_argument("--slurm-user", default="gaozh")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    status = build_status(
        data_root=args.data_root,
        cninfo_panel=args.cninfo_panel,
        report=args.report,
        include_slurm=args.include_slurm,
        slurm_user=args.slurm_user,
    )
    payload = json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()
