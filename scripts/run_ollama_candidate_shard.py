"""Create resumable Qwen embeddings for the strongest text candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.ollama_client import ollama_embed


CANDIDATES = {
    "plain": "text_plain",
    "short": "text_input_2_short",
    "masked_short": "text_input_5_masked_short",
}


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            yield record


def source_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": stat.st_size, "sha256": digest.hexdigest()}


def valid_output(path: Path, summary_path: Path, expected: dict[str, object]) -> bool:
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        matrix = np.load(path, mmap_mode="r")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    keys = ("model", "candidate", "text_column", "max_tokens", "rows", "source_sha256")
    return matrix.ndim == 2 and matrix.shape[0] == expected["rows"] and all(
        summary.get(key) == expected.get(key) for key in keys
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", default="short,masked_short,plain")
    parser.add_argument("--model", default="qwen3-embedding:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-rows", type=int, default=0, help="benchmark only; 0 processes the whole shard")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.batch_size < 1 or args.max_rows < 0:
        raise ValueError("max-tokens and batch-size must be positive; max-rows cannot be negative")
    candidates = [value.strip() for value in args.candidates.split(",") if value.strip()]
    unknown = sorted(set(candidates) - set(CANDIDATES))
    if unknown:
        raise ValueError(f"unknown candidates: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint(args.input)
    rows = list(iter_rows(args.input))
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError(f"no rows found in {args.input}")

    metadata_path = args.output_dir / "metadata.parquet"
    if not metadata_path.exists():
        metadata = pd.DataFrame([
            {
                "row_index": row.get("row_index"),
                "document_id": row.get("document_id"),
                "stock_id_alignment": row.get("stock_id_alignment"),
                "announcement_date_alignment": row.get("announcement_date_alignment"),
            }
            for row in rows
        ])
        temporary = metadata_path.with_suffix(metadata_path.suffix + f".tmp.{os.getpid()}")
        metadata.to_parquet(temporary, index=False)
        os.replace(temporary, metadata_path)

    for candidate in candidates:
        text_column = CANDIDATES[candidate]
        output = args.output_dir / f"{candidate}.npy"
        summary_path = args.output_dir / f"{candidate}.summary.json"
        expected = {
            "model": args.model,
            "candidate": candidate,
            "text_column": text_column,
            "max_tokens": args.max_tokens,
            "rows": len(rows),
            "source_sha256": fingerprint["sha256"],
        }
        if valid_output(output, summary_path, expected):
            print(json.dumps({"status": "skipped_complete", "output": str(output), **expected}), flush=True)
            continue

        temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
        handle: np.memmap | None = None
        started = time.monotonic()
        for start in range(0, len(rows), args.batch_size):
            texts = [str(row.get(text_column, "") or "") for row in rows[start : start + args.batch_size]]
            matrix = np.asarray(
                ollama_embed(
                    texts,
                    model=args.model,
                    base_url=args.base_url,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                ),
                dtype=np.float32,
            )
            if matrix.ndim != 2 or matrix.shape[0] != len(texts):
                raise RuntimeError(f"unexpected Ollama shape {matrix.shape} for batch size {len(texts)}")
            if handle is None:
                handle = np.lib.format.open_memmap(
                    temporary, mode="w+", dtype=np.float32, shape=(len(rows), matrix.shape[1])
                )
            handle[start : start + len(texts)] = matrix
            if start == 0 or (start // args.batch_size + 1) % 50 == 0:
                handle.flush()
                print(json.dumps({
                    "status": "progress", "candidate": candidate,
                    "rows_done": start + len(texts), "rows_total": len(rows),
                    "elapsed_seconds": time.monotonic() - started,
                }), flush=True)
        assert handle is not None
        handle.flush()
        dimensions = int(handle.shape[1])
        del handle
        os.replace(temporary, output)
        elapsed = time.monotonic() - started
        summary = {
            **expected,
            "dimensions": dimensions,
            "batch_size": args.batch_size,
            "elapsed_seconds": elapsed,
            "rows_per_second": len(rows) / elapsed,
            "source": fingerprint,
            "output": str(output),
        }
        summary_temporary = summary_path.with_suffix(summary_path.suffix + f".tmp.{os.getpid()}")
        summary_temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(summary_temporary, summary_path)
        print(json.dumps({"status": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
