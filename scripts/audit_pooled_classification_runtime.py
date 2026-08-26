"""Map historical pooled-classification Slurm accounting to manifest cells."""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARRAY_TASK = re.compile(r"^(?P<array>\d+)_(?P<task>\d+)$")
MANIFEST = re.compile(r"(?:^|,)MANIFEST=(?P<path>[^,\s]+)")


def memory_kib(value: str) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", value.strip(), re.IGNORECASE)
    if not match:
        return None
    scales = {"": 1 / 1024, "K": 1, "M": 1024, "G": 1024 ** 2, "T": 1024 ** 3, "P": 1024 ** 4}
    return int(float(match.group(1)) * scales[match.group(2).upper()])


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def accounting(start: str) -> list[list[str]]:
    command = [
        "sacct", "-S", start, "-u", os.environ.get("USER", ""), "--name", "pooled-cls",
        "-n", "-P", "--format=JobID,State,ElapsedRaw,MaxRSS,SubmitLine%1000",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return [line.split("|", 4) for line in completed.stdout.splitlines() if line.strip()]


def manifest_row(path: Path, task_id: int) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [row for row in rows if int(row["task_id"]) == task_id]
    if len(matches) != 1:
        raise ValueError(f"expected one task {task_id} in {path}; found {len(matches)}")
    return matches[0]


def audit(start: str) -> list[dict[str, object]]:
    raw = accounting(start)
    max_rss_by_task: dict[str, int] = {}
    for job_id, _state, _elapsed, max_rss, _submit in raw:
        if not job_id.endswith(".batch"):
            continue
        parsed = memory_kib(max_rss)
        if parsed is not None:
            max_rss_by_task[job_id.removesuffix(".batch")] = parsed

    rows: list[dict[str, object]] = []
    for job_id, state, elapsed, _max_rss, submit_line in raw:
        match = ARRAY_TASK.fullmatch(job_id)
        manifest_match = MANIFEST.search(submit_line)
        if not match or not manifest_match or not state.startswith("COMPLETED"):
            continue
        relative_manifest = Path(manifest_match.group("path"))
        manifest_path = relative_manifest if relative_manifest.is_absolute() else ROOT / relative_manifest
        if not manifest_path.is_file():
            continue
        cell = manifest_row(manifest_path, int(match.group("task")))
        rows.append({
            "job_id": job_id,
            "manifest": str(manifest_path.relative_to(ROOT)),
            **cell,
            "elapsed_seconds": int(elapsed),
            "max_rss_kib": max_rss_by_task.get(job_id),
        })
    return rows


def write(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "job_id", "manifest", "task_id", "phase", "embedding_root", "model", "variant",
        "feature", "classifier", "reducer", "components", "output", "elapsed_seconds", "max_rss_kib",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["classifier"]), str(row["reducer"]))].append(row)
    lines = [
        "# Historical pooled-classification runtime audit",
        "",
        "Slurm `ElapsedRaw` is mapped to the exact array cell through its persisted TSV manifest. "
        "These measurements describe completed historical jobs; they are not scheduler limits.",
        "",
        "| Classifier | Reducer | Jobs | Median minutes | P90 minutes | Max hours | Peak RSS GiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (classifier, reducer), group in sorted(grouped.items()):
        elapsed = [int(row["elapsed_seconds"]) for row in group]
        rss = [int(row["max_rss_kib"]) for row in group if row.get("max_rss_kib") is not None]
        lines.append(
            f"| `{classifier}` | `{reducer}` | {len(group)} | "
            f"{statistics.median(elapsed) / 60:.2f} | {percentile(elapsed, 0.9) / 60:.2f} | "
            f"{max(elapsed) / 3600:.2f} | {max(rss) / 1024 ** 2:.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Calibrated Linear SVM without PCA is the dominant long-running cell; PCA-128 makes it much cheaper.",
        "- PCA cells in these manifests include embedding loading, two training-window PCA fits, tuning and final fitting.",
        "- Future reports record phase-level runtime directly, so subsequent estimates need not rely only on Slurm wall time.",
        "",
    ])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-08-01")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "references" / "pooled_classification_runtime_audit.csv",
    )
    args = parser.parse_args()
    rows = audit(args.start)
    if not rows:
        raise ValueError("no completed pooled classification array cells could be mapped")
    write(rows, args.output)
    print(f"rows={len(rows)} csv={args.output} markdown={args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()