"""Create reproducible records before Slurm or other long-running submissions."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = ROOT / "task_records"
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
ENV_PREFIXES = ("SLURM_", "CUDA_", "OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_", "HF_", "TRANSFORMERS_")
ENV_NAMES = {"PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"}
PACKAGE_NAMES = ("numpy", "pandas", "scikit-learn", "torch", "transformers", "pyarrow")
PARSABLE_JOB_ID_RE = re.compile(r"^(\d+)(?:_[0-9,+%\-]+)?(?:;[^\s]+)?$")
SUBMITTED_JOB_ID_RE = re.compile(r"^Submitted batch job (\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_capture(command: Sequence[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        list(command), cwd=ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout


def git_output(*args: str, check: bool = True) -> str:
    return run_capture(("git", *args), check=check)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug[:48] or "task"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact(name: str, value: str) -> str:
    return "<redacted>" if any(marker in name.upper() for marker in SECRET_MARKERS) else value


def selected_environment() -> dict[str, str]:
    result = {}
    for name, value in sorted(os.environ.items()):
        if name in ENV_NAMES or name.startswith(ENV_PREFIXES):
            result[name] = redact(name, value)
    return result


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def parse_sbatch_directives(path: Path | None) -> dict[str, list[str]]:
    directives: dict[str, list[str]] = {}
    if path is None or not path.is_file():
        return directives
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("#SBATCH"):
                continue
            value = stripped[len("#SBATCH"):].strip()
            key = value.split("=", 1)[0].split(None, 1)[0]
            directives.setdefault(key, []).append(value)
    return directives


def command_file_paths(command: Sequence[str], related: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for raw in (*command, *related):
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        try:
            candidate.relative_to(ROOT)
        except ValueError:
            continue
        if candidate.is_file():
            paths.add(candidate.resolve())
    return sorted(paths)


def find_sbatch_script(command: Sequence[str]) -> Path | None:
    for raw in reversed(command):
        if raw.endswith(".sbatch"):
            path = Path(raw)
            return (path if path.is_absolute() else ROOT / path).resolve()
    return None


def write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def snapshot_code(task_dir: Path, relevant_files: Sequence[Path]) -> dict[str, Any]:
    code_dir = task_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=False)
    status = git_output("status", "--short", "--untracked-files=all")
    (code_dir / "git-status.txt").write_text(status, encoding="utf-8")
    # Include both unstaged and staged tracked changes, preserving binary patches.
    unstaged = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"], cwd=ROOT,
        check=True, stdout=subprocess.PIPE,
    ).stdout
    staged = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "--cached"], cwd=ROOT,
        check=True, stdout=subprocess.PIPE,
    ).stdout
    patch = code_dir / "tracked-dirty.patch"
    patch.write_bytes(b"# unstaged tracked changes\n" + unstaged + b"\n# staged tracked changes\n" + staged)

    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, stdout=subprocess.PIPE,
    ).stdout
    untracked = [
        item.decode("utf-8", errors="surrogateescape")
        for item in untracked_raw.split(b"\0") if item
    ]
    untracked = sorted(path for path in untracked if not path.startswith("task_records/"))
    (code_dir / "untracked-files.txt").write_text("".join(f"{path}\n" for path in untracked), encoding="utf-8")
    archive = code_dir / "untracked.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for relative in untracked:
            source = ROOT / relative
            if source.is_file() or source.is_symlink():
                tar.add(source, arcname=relative, recursive=False)

    hashes = {}
    for path in relevant_files:
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    for path in (code_dir / "git-status.txt", patch, code_dir / "untracked-files.txt", archive):
        hashes[str(path.relative_to(task_dir))] = sha256(path)
    write_atomic_json(code_dir / "file-hashes.json", hashes)
    return {"untracked_file_count": len(untracked), "file_hashes": hashes}


def reference_snapshot(
    record_root: Path, source_task_id: str, relevant_files: Sequence[Path],
) -> dict[str, Any]:
    """Reference an existing full snapshot while hashing this task's related files."""
    source_dir = record_root / source_task_id
    source_manifest_path = source_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError(f"snapshot source task does not exist: {source_task_id}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_snapshot = source_manifest.get("snapshot", {})
    source_hashes = source_snapshot.get("file_hashes", {})
    archive_hash = source_hashes.get("code/untracked.tar.gz")
    archive = source_dir / "code" / "untracked.tar.gz"
    if not archive_hash or not archive.is_file():
        raise ValueError(f"source task has no complete code archive: {source_task_id}")
    if sha256(archive) != archive_hash:
        raise ValueError(f"source task archive hash mismatch: {source_task_id}")
    return {
        "reused_from_task": source_task_id,
        "source_archive": str(archive),
        "source_archive_sha256": archive_hash,
        "source_git": source_manifest.get("git"),
        "related_file_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in relevant_files
        },
    }


def create_task(args: argparse.Namespace, command: Sequence[str]) -> tuple[str, Path, dict[str, Any]]:
    record_root = Path(args.record_root).resolve()
    record_root.mkdir(parents=True, exist_ok=True)
    task_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{slugify(args.name)}-{os.getpid()}"
    task_dir = record_root / task_id
    task_dir.mkdir(parents=False, exist_ok=False)
    related_files = command_file_paths(command, args.related_file)
    sbatch_script = find_sbatch_script(command)
    snapshot_from_task = getattr(args, "snapshot_from_task", None)
    snapshot = (
        reference_snapshot(record_root, snapshot_from_task, related_files)
        if snapshot_from_task else snapshot_code(task_dir, related_files)
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task_id": task_id,
        "name": args.name,
        "purpose": args.purpose,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "recorded",
        "repository": str(ROOT),
        "git": {
            "head": git_output("rev-parse", "HEAD").strip(),
            "branch": git_output("branch", "--show-current").strip(),
            "dirty": bool(git_output("status", "--porcelain")),
        },
        "snapshot": snapshot,
        "command": list(command),
        "command_shell": shlex.join(command),
        "sbatch_script": str(sbatch_script.relative_to(ROOT)) if sbatch_script else None,
        "sbatch_directives": parse_sbatch_directives(sbatch_script),
        "inputs": list(args.input),
        "outputs": list(args.output),
        "seed": args.seed,
        "tags": list(args.tag),
        "related_files": [str(path.relative_to(ROOT)) for path in related_files],
        "runtime": {
            "executable": sys.executable,
            "python": sys.version,
            "packages": package_versions(),
            "environment": selected_environment(),
            "hostname": os.uname().nodename,
        },
        "submissions": [],
        "events": [{"at": utc_now(), "type": "record_created"}],
        "notes": [],
    }
    write_atomic_json(task_dir / "manifest.json", manifest)
    return task_id, task_dir, manifest


def locked_manifest(task_dir: Path):
    lock_path = task_dir / ".lock"
    lock_path.touch(exist_ok=True)
    handle = lock_path.open("r+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def update_manifest(task_dir: Path, callback) -> dict[str, Any]:
    lock = locked_manifest(task_dir)
    try:
        path = task_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        callback(manifest)
        manifest["updated_at"] = utc_now()
        write_atomic_json(path, manifest)
        return manifest
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def extract_job_ids(text: str) -> list[str]:
    job_ids = []
    for line in text.splitlines():
        stripped = line.strip()
        match = PARSABLE_JOB_ID_RE.fullmatch(stripped) or SUBMITTED_JOB_ID_RE.fullmatch(stripped)
        if match:
            job_ids.append(match.group(1))
    return list(dict.fromkeys(job_ids))


def run_recorded_command(task_id: str, task_dir: Path, command: Sequence[str]) -> int:
    environment = os.environ.copy()
    environment.update({"TASK_RECORD_ID": task_id, "TASK_RECORD_ROOT": str(task_dir.parent)})
    def started(manifest):
        manifest["status"] = "submitting"
        manifest["events"].append({"at": utc_now(), "type": "command_started"})
    update_manifest(task_dir, started)
    completed = subprocess.run(list(command), cwd=ROOT, env=environment, text=True, capture_output=True)
    (task_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (task_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    job_ids = extract_job_ids(completed.stdout)
    def finished(manifest):
        manifest["status"] = "submitted" if completed.returncode == 0 else "submission_failed"
        manifest["top_level_returncode"] = completed.returncode
        manifest["top_level_job_ids"] = job_ids
        manifest["events"].append({
            "at": utc_now(), "type": "command_finished",
            "returncode": completed.returncode, "job_ids": job_ids,
        })
    update_manifest(task_dir, finished)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    print(f"TASK_RECORD_ID={task_id}", file=sys.stderr)
    return completed.returncode


def child_submit(args: argparse.Namespace, command: Sequence[str]) -> int:
    task_dir = Path(args.record_root).resolve() / args.task_id
    if not (task_dir / "manifest.json").is_file():
        raise SystemExit(f"task record not found: {task_dir}")
    submission_id = f"attempt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
    def before(manifest):
        manifest["submissions"].append({
            "submission_id": submission_id, "started_at": utc_now(),
            "command": list(command), "command_shell": shlex.join(command),
            "status": "submitting", "replaces_job_id": args.replaces_job_id or None,
        })
        manifest["events"].append({"at": utc_now(), "type": "child_submission_started", "submission_id": submission_id})
    update_manifest(task_dir, before)
    environment = os.environ.copy()
    environment.update({"TASK_RECORD_ID": args.task_id, "TASK_RECORD_ROOT": str(task_dir.parent)})
    completed = subprocess.run(list(command), cwd=ROOT, env=environment, text=True, capture_output=True)
    job_ids = extract_job_ids(completed.stdout)
    def after(manifest):
        item = next(item for item in manifest["submissions"] if item["submission_id"] == submission_id)
        item.update({
            "finished_at": utc_now(), "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "job_ids": job_ids,
            "status": "submitted" if completed.returncode == 0 else "submission_failed",
        })
        manifest["events"].append({"at": utc_now(), "type": "child_submission_finished", "submission_id": submission_id, "job_ids": job_ids})
    update_manifest(task_dir, after)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


def sync_task(args: argparse.Namespace) -> int:
    task_dir = Path(args.record_root).resolve() / args.task_id
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    job_ids = list(manifest.get("top_level_job_ids", []))
    for item in manifest.get("submissions", []):
        job_ids.extend(item.get("job_ids", []))
    job_ids = list(dict.fromkeys(job_ids))
    if not job_ids:
        raise SystemExit("task contains no job IDs")
    completed = subprocess.run(
        ["sacct", "-j", ",".join(job_ids), "-X", "--json"], cwd=ROOT,
        text=True, capture_output=True,
    )
    def apply(manifest):
        manifest["scheduler_sync"] = {
            "at": utc_now(), "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
        }
        manifest["events"].append({"at": utc_now(), "type": "scheduler_synced", "job_ids": job_ids})
    update_manifest(task_dir, apply)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


def close_task(args: argparse.Namespace) -> int:
    task_dir = Path(args.record_root).resolve() / args.task_id
    def close(manifest):
        manifest["status"] = args.status
        if args.note:
            manifest["notes"].append({"at": utc_now(), "text": args.note})
        manifest["events"].append({"at": utc_now(), "type": "task_closed", "status": args.status})
    update_manifest(task_dir, close)
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record-root", default=str(DEFAULT_RECORD_ROOT))


def command_after_separator(values: Sequence[str]) -> list[str]:
    command = list(values)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("a command is required after --")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="snapshot, then run a top-level submission command")
    add_common(run_parser)
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--purpose", required=True)
    run_parser.add_argument("--input", action="append", default=[])
    run_parser.add_argument("--output", action="append", default=[])
    run_parser.add_argument("--related-file", action="append", default=[])
    run_parser.add_argument("--tag", action="append", default=[])
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument(
        "--snapshot-from-task", default="",
        help="Reference a prior complete code snapshot instead of archiving it again.",
    )
    run_parser.add_argument("command", nargs=argparse.REMAINDER)

    child_parser = subparsers.add_parser("child-submit", help="record and run a child submission")
    add_common(child_parser)
    child_parser.add_argument("--task-id", required=True)
    child_parser.add_argument("--replaces-job-id", default="")
    child_parser.add_argument("command", nargs=argparse.REMAINDER)

    sync_parser = subparsers.add_parser("sync", help="capture scheduler status for all recorded job IDs")
    add_common(sync_parser)
    sync_parser.add_argument("--task-id", required=True)

    close_parser = subparsers.add_parser("close", help="close a task after output audit")
    add_common(close_parser)
    close_parser.add_argument("--task-id", required=True)
    close_parser.add_argument("--status", choices=("completed_audited", "failed_audited", "cancelled"), required=True)
    close_parser.add_argument("--note", default="")

    args = parser.parse_args()
    if args.action == "run":
        command = command_after_separator(args.command)
        task_id, task_dir, _ = create_task(args, command)
        raise SystemExit(run_recorded_command(task_id, task_dir, command))
    if args.action == "child-submit":
        raise SystemExit(child_submit(args, command_after_separator(args.command)))
    if args.action == "sync":
        raise SystemExit(sync_task(args))
    if args.action == "close":
        raise SystemExit(close_task(args))


if __name__ == "__main__":
    main()
