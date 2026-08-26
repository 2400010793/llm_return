"""Content-addressed experiment bundles for reusable research models."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import joblib


def file_fingerprint(path: Path, *, hash_content: bool = False) -> dict[str, Any]:
    """Return a stable input fingerprint without repeatedly hashing huge matrices."""
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_content:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def experiment_id(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_input_fingerprints(
    panel: Path,
    parts: Iterable[Path],
    *,
    include_prompt_tokens: bool = False,
) -> dict[str, Any]:
    """Fingerprint control files exactly and large NPZ payloads by immutable stat facts.

    Summary and metadata content hashes protect identity and row alignment. Large
    matrices use size plus nanosecond mtime to keep Slurm array startup cheap;
    task records retain the stronger provenance snapshot for submitted runs.
    Prompt-token runs additionally fingerprint the separate memory-mapped
    token arrays and their small token metadata files.
    """
    part_rows = []
    for part in parts:
        row = {
            "directory": str(part),
            "summary": file_fingerprint(part / "summary.json", hash_content=True),
            "metadata": file_fingerprint(part / "metadata.jsonl", hash_content=True),
            "matrix": file_fingerprint(part / "short_pooling.npz"),
        }
        if include_prompt_tokens:
            prompt_files = {}
            for name, hash_content in (
                ("prompt_token_embeddings.npy", False),
                ("prompt_input_ids.npy", True),
                ("prompt_tokens.json", True),
            ):
                path = part / name
                if path.is_file():
                    prompt_files[name] = file_fingerprint(path, hash_content=hash_content)
            if prompt_files:
                row["prompt_files"] = prompt_files
        part_rows.append(row)
    return {
        "panel": file_fingerprint(panel, hash_content=True),
        "embedding_parts": part_rows,
    }


def completed_bundle_matches(bundle: Path, expected_id: str) -> bool:
    marker = bundle / "COMPLETED"
    spec_path = bundle / "spec.json"
    manifest_path = bundle / "manifest.json"
    if not marker.is_file() or not spec_path.is_file() or not manifest_path.is_file():
        return False
    try:
        stored = json.loads(spec_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        marker.read_text(encoding="utf-8").strip() != expected_id
        or stored.get("experiment_id") != expected_id
        or manifest.get("experiment_id") != expected_id
    ):
        return False
    fingerprints = {
        bundle / relative: fingerprint
        for relative, fingerprint in manifest.get("files", {}).items()
    }
    fingerprints.update({
        Path(fingerprint["path"]): fingerprint
        for fingerprint in manifest.get("external_outputs", {}).values()
    })
    for path, expected in fingerprints.items():
        if not path.is_file():
            return False
        actual = file_fingerprint(path, hash_content="sha256" in expected)
        if any(actual.get(key) != value for key, value in expected.items() if key != "path"):
            return False
    return True


def acquire_bundle_lock(bundle: Path, *, stale_after_seconds: float = 86_400) -> Path:
    """Claim the single-writer lock for a bundle, recovering only stale locks."""
    bundle.mkdir(parents=True, exist_ok=True)
    lock = bundle / ".writer.lock"
    for attempt in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - lock.stat().st_mtime
            if attempt == 0 and age > stale_after_seconds:
                lock.unlink()
                continue
            raise RuntimeError(f"artifact bundle is already being written: {bundle}")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        return lock
    raise RuntimeError(f"could not lock artifact bundle: {bundle}")


def release_bundle_lock(lock: Path) -> None:
    lock.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_joblib(path: Path, value: Any, *, attempts: int = 3) -> None:
    """Serialize locally, then retry a checked atomic publish to shared storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    if not scratch_root.is_dir():
        scratch_root = path.parent
    with tempfile.NamedTemporaryFile(
        dir=scratch_root, prefix=f".{path.name}.", suffix=".joblib", delete=False,
    ) as handle:
        scratch = Path(handle.name)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        joblib.dump(value, scratch)
        expected_size = scratch.stat().st_size
        for attempt in range(1, attempts + 1):
            try:
                shutil.copyfile(scratch, temporary)
                if temporary.stat().st_size != expected_size:
                    raise OSError(
                        f"incomplete joblib copy: {temporary.stat().st_size} != {expected_size}"
                    )
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                return
            except OSError:
                temporary.unlink(missing_ok=True)
                if attempt == attempts:
                    raise
                time.sleep(attempt)
    finally:
        scratch.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def write_completed(bundle: Path, expected_id: str) -> None:
    temporary = bundle / f"COMPLETED.tmp.{os.getpid()}"
    temporary.write_text(expected_id + "\n", encoding="utf-8")
    os.replace(temporary, bundle / "COMPLETED")
