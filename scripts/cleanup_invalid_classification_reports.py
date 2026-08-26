"""Delete confirmed-invalid classification reports with a hash-only audit trail.

The command is a dry run unless ``--apply`` is provided. It never deletes
embeddings, panels, logs, trusted reports, or Slurm outputs outside the two
explicitly verified invalid report directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVALID_DIRECTORIES = {
    Path("reports/classification/time_aligned_prompt_embeddings"): (
        "invalid_alignment",
        "Full-panel embeddings were position-indexed by the legacy panel without metadata-key alignment.",
    ),
    Path("reports/classification/transformers_time_aligned"): (
        "mislabelled_representation",
        "Transformer-named reports contain mixed representation families and cannot support transformer conclusions.",
    ),
}
DERIVED_OUTPUTS = (
    Path("reports/classification/time_aligned_model_matrix.csv"),
    Path("reports/classification/time_aligned_model_matrix.md"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative_directory, (status, reason) in INVALID_DIRECTORIES.items():
        directory = ROOT / relative_directory
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                raise ValueError(f"unexpected non-file in invalid report directory: {path}")
            if path.suffix != ".json":
                raise ValueError(f"refusing to delete unexpected non-JSON artifact: {path}")
            rows.append({
                "path": str(path.relative_to(ROOT)),
                "status": status,
                "reason": reason,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return rows


def write_manifest(rows: list[dict[str, object]], output: Path, *, applied: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    deleted_at = datetime.now(timezone.utc).isoformat() if applied else ""
    fields = ("path", "status", "reason", "size_bytes", "sha256", "deleted_at_utc")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "deleted_at_utc": deleted_at})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete files after writing the manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "references" / "classification_report_cleanup_manifest.csv",
    )
    args = parser.parse_args()
    rows = collect()
    if not rows and args.manifest.exists():
        print(f"files=0 applied=False already_clean=True manifest={args.manifest}")
        return
    if args.apply and len(rows) != 106:
        raise ValueError(f"expected exactly 106 confirmed-invalid report files; found {len(rows)}")
    write_manifest(rows, args.manifest, applied=args.apply)
    if args.apply:
        for row in rows:
            (ROOT / str(row["path"])).unlink()
        for relative_directory in INVALID_DIRECTORIES:
            directory = ROOT / relative_directory
            if directory.exists():
                directory.rmdir()
        for relative in DERIVED_OUTPUTS:
            path = ROOT / relative
            if path.exists():
                path.unlink()
    print(f"files={len(rows)} applied={args.apply} manifest={args.manifest}")


if __name__ == "__main__":
    main()