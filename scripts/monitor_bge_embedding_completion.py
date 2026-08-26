"""Recover failed BGE array tasks and release only fully validated outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge_incremental_pooled_embeddings import _validate_range


TERMINAL_FAILURES = {
    "BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
}
REQUIRED_FILES = ("summary.json", "metadata.jsonl", "short_pooling.npz")


def parse_job_variant(value: str) -> tuple[str, str]:
    job, separator, variant = value.partition(":")
    if not separator or not job.isdigit() or variant not in {"short", "masked_short"}:
        raise argparse.ArgumentTypeError("expected JOB_ID:short or JOB_ID:masked_short")
    return job, variant


def output_directory(root: Path, task_id: int, variant: str) -> Path:
    return root / f"shard-{task_id // 2}" / "bge_m3" / variant


def output_complete(directory: Path) -> bool:
    if not all((directory / name).is_file() and (directory / name).stat().st_size > 0
               for name in REQUIRED_FILES):
        return False
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        rows = int(summary["rows"])
        outputs = summary["outputs"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    required_outputs = {
        "prompt_token_embeddings", "prompt_mean", "title_mean", "body_mean",
        "title_body_mean", "full_mean",
    }
    if rows < 1 or not required_outputs.issubset(outputs):
        return False
    try:
        with zipfile.ZipFile(directory / "short_pooling.npz") as archive:
            members = set(archive.namelist())
            required_members = {f"{name}.npy" for name in required_outputs}
            if not required_members.issubset(members):
                return False
            for name in required_members:
                with archive.open(name) as member:
                    member.read(1)
    except (OSError, RuntimeError, zipfile.BadZipFile):
        return False
    return True


def task_state(job_id: str, task_id: int) -> str:
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", f"{job_id}_{task_id}", "--format=State", "-P"],
        check=True, text=True, capture_output=True,
    )
    states = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
    return states[0] if states else "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-variant", action="append", type=parse_job_variant, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--row-start", type=int, required=True)
    parser.add_argument("--row-end", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-requeues", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.max_requeues < 1:
        raise ValueError("poll-seconds and max-requeues must be positive")

    retries: dict[str, int] = {}
    while True:
        incomplete: list[tuple[str, int, str, Path]] = []
        for job_id, variant in args.job_variant:
            for task_id in range(1, 64, 2):
                directory = output_directory(args.output_root, task_id, variant)
                if not output_complete(directory):
                    incomplete.append((job_id, task_id, variant, directory))
        if not incomplete:
            break

        for job_id, task_id, variant, directory in incomplete:
            state = task_state(job_id, task_id)
            if state not in TERMINAL_FAILURES and state != "COMPLETED":
                continue
            key = f"{job_id}_{task_id}"
            retries[key] = retries.get(key, 0) + 1
            if retries[key] > args.max_requeues:
                raise RuntimeError(
                    f"{key} still lacks complete {variant} output after "
                    f"{args.max_requeues} requeues: {directory}"
                )
            subprocess.run(["scontrol", "requeue", key], check=True)
            print(json.dumps({"requeued": key, "state": state, "attempt": retries[key]}), flush=True)
        print(json.dumps({"incomplete": len(incomplete), "retries": retries}), flush=True)
        time.sleep(args.poll_seconds)

    variants = tuple(variant for _job, variant in args.job_variant)
    _validate_range(args.output_root, ("bge_m3",), variants, args.row_start, args.row_end)
    report = {
        "output_root": str(args.output_root),
        "model": "bge_m3",
        "variants": list(variants),
        "row_start": args.row_start,
        "row_end": args.row_end,
        "rows": args.row_end - args.row_start + 1,
        "requeues": retries,
        "complete": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
