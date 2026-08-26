"""Persist one scheduler and artifact snapshot for the active CNINFO pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path


TASK_RE = re.compile(r"^(?P<job>\d+)_(?P<task>\d+)$")
CHILD_JOB_RE = re.compile(r"submitted all-pooling classification tasks=\d+ job=(\d+)")
FAILURE_STATES = {"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "CANCELLED"}


def run(*command: str) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def job_state(job_id: str) -> str:
    queued_result = subprocess.run(
        ("squeue", "-h", "-j", job_id, "-o", "%T"),
        check=False, text=True, capture_output=True,
    )
    queued = queued_result.stdout.strip().splitlines()
    if queued:
        return queued[0]
    values = [line.strip().split()[0] for line in run(
        "sacct", "-n", "-X", "-j", job_id, "--format=State", "-P",
    ).splitlines() if line.strip()]
    return values[0] if values else "UNKNOWN"


def array_tasks(job_id: str, *, parity: int | None = None) -> dict[str, object]:
    output = run(
        "sacct", "-n", "-X", "-j", job_id,
        "--format=JobID,State,Elapsed,MaxRSS,ExitCode", "-P",
    )
    rows = []
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 5 or not (match := TASK_RE.match(fields[0])):
            continue
        task = int(match.group("task"))
        if parity is not None and task % 2 != parity:
            continue
        state = fields[1].split()[0]
        rows.append({
            "task": task, "state": state, "elapsed": fields[2],
            "max_rss": fields[3], "exit_code": fields[4],
        })
    counts = Counter(row["state"] for row in rows)
    failures = [row for row in rows if row["state"] in FAILURE_STATES]
    return {"counts": dict(sorted(counts.items())), "failures": failures, "tasks": len(rows)}


def discover_child_job(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    matches = CHILD_JOB_RE.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    return matches[-1] if matches else None


def last_json_line(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def artifact_counts(root: Path, model: str) -> dict[str, int]:
    return {
        "reports": len(list(root.glob(f"{model}_*.json"))),
        "completed_bundles": len(list(root.glob(f"{model}_*.artifacts/COMPLETED"))),
    }


def write_latest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latest", type=Path, required=True)
    parser.add_argument("--roberta-classification-job", default="4330745")
    parser.add_argument("--bge-short-job", default="4330365")
    parser.add_argument("--bge-masked-job", default="4330639")
    parser.add_argument("--bge-barrier-job", default="4330814")
    parser.add_argument("--bge-merge-job", default="4330367")
    parser.add_argument("--bge-launcher-job", default="4330738")
    parser.add_argument("--bge-recovery-job", default="4333420")
    args = parser.parse_args()

    report_root = Path("reports/classification/pooled_embeddings/all_pooling")
    bge_child = discover_child_job(
        Path(f"logs/slurm_cninfo_all_pool_cls_launcher/{args.bge_launcher_job}.out")
    )
    roberta = array_tasks(args.roberta_classification_job)
    roberta["expected_tasks"] = 182
    bge_classification = None
    if bge_child:
        bge_classification = array_tasks(bge_child)
        bge_classification.update({"job_id": bge_child, "expected_tasks": 182})
    payload = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "roberta_classification": {
            "job_id": args.roberta_classification_job,
            **roberta,
            **artifact_counts(report_root, "roberta"),
        },
        "bge_embeddings": {
            "short": {"job_id": args.bge_short_job, **array_tasks(args.bge_short_job, parity=1)},
            "masked_short": {
                "job_id": args.bge_masked_job,
                **array_tasks(args.bge_masked_job, parity=1),
            },
            "barrier": {
                "job_id": args.bge_barrier_job,
                "state": job_state(args.bge_barrier_job),
                "latest": last_json_line(
                    Path(f"logs/slurm_bge_embed_barrier/{args.bge_barrier_job}.out")
                ),
            },
        },
        "bge_pipeline": {
            "merge": {"job_id": args.bge_merge_job, "state": job_state(args.bge_merge_job)},
            "launcher": {
                "job_id": args.bge_launcher_job,
                "state": job_state(args.bge_launcher_job),
            },
            "classification": bge_classification,
            "classification_recovery": {
                "job_id": args.bge_recovery_job,
                **array_tasks(args.bge_recovery_job),
            },
            **artifact_counts(report_root, "bge_m3"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    write_latest(args.latest, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
