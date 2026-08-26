"""Build a leakage-safe 2010--2026 CNINFO JSONL input with stable row IDs.

The current 2018--2026 input is kept byte-order compatible with the existing
panel so its pooled embeddings remain reusable. Historical records are
validated, sorted deterministically, and appended after the current rows.
"""

from __future__ import annotations

import argparse
import contextlib
import heapq
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def stock_ids_from_panel(path: Path) -> set[str]:
    frame = pd.read_parquet(path, columns=["stock_id"])
    values = frame["stock_id"].astype(str).str.strip().str.zfill(6)
    return set(values[values.str.fullmatch(r"\d{6}")])


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def stock_id(record: dict[str, Any]) -> str:
    return str(record.get("stock_id", "") or "").strip().zfill(6)


def publication_time(record: dict[str, Any]) -> pd.Timestamp | None:
    value = record.get("published_at") or record.get("announcement_date")
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.tz_convert(None)


def text_value(record: dict[str, Any]) -> str:
    return str(record.get("text_model", record.get("text", "")) or "")


def stable_document_id(record: dict[str, Any]) -> str:
    supplied = str(record.get("document_id", "") or "").strip()
    if supplied:
        return supplied
    fields = (
        stock_id(record),
        str(record.get("url", "") or ""),
        str(record.get("announcement_date", "") or ""),
        str(record.get("published_at", "") or ""),
        str(record.get("title_clean_final", record.get("title", "")) or ""),
        str(record.get("text_hash", "") or ""),
    )
    return "cninfo_" + hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:32]


def validate_record(
    record: dict[str, Any], allowed_stocks: set[str], *, period: str,
) -> tuple[dict[str, Any] | None, str | None]:
    sid = stock_id(record)
    if sid not in allowed_stocks:
        return None, "outside_stock_universe"
    if record.get("content_type") != "announcement":
        return None, "non_announcement"
    if "stock_relation" in record and record.get("stock_relation") != "direct":
        return None, "non_direct_relation"
    timestamp = publication_time(record)
    if timestamp is None:
        return None, "invalid_publication_time"
    text = text_value(record)
    if not text.strip():
        return None, "empty_text_model"
    result = dict(record)
    result["stock_id"] = sid
    result["published_at"] = timestamp.isoformat()
    result["document_id"] = stable_document_id(result)
    result["text_model"] = text
    result["source_period"] = period
    return result, None


def normalize_frozen_record(record: dict[str, Any], *, period: str) -> dict[str, Any]:
    """Normalize one row without changing the frozen current-input row set."""
    timestamp = publication_time(record)
    if timestamp is None:
        raise ValueError("frozen current record has no valid publication time")
    result = dict(record)
    result["stock_id"] = stock_id(record)
    result["published_at"] = timestamp.isoformat()
    result["document_id"] = stable_document_id(result)
    result["text_model"] = text_value(record)
    result["source_period"] = period
    return result


def load_existing_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["row_index", "stock_id", "published_at"])
    expected = pd.Series(range(1, len(frame) + 1), dtype="int64")
    actual = pd.to_numeric(frame["row_index"], errors="raise").astype("int64")
    if not actual.reset_index(drop=True).equals(expected):
        raise ValueError("existing panel row_index must be the contiguous sequence 1..N")
    return frame.reset_index(drop=True)


HistoricalKey = tuple[str, int, str]


def historical_sort_key(record: dict[str, Any]) -> HistoricalKey:
    timestamp = publication_time(record)
    if timestamp is None:
        raise ValueError("validated historical record has no publication time")
    return (stock_id(record), int(timestamp.value), str(record["document_id"]))


def write_sorted_chunk(
    path: Path, records: list[tuple[HistoricalKey, dict[str, Any]]]
) -> None:
    """Persist one bounded sorted run for the disk-backed historical merge."""
    records.sort(key=lambda item: item[0])
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for key, record in records:
                key_json = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
                record_json = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                handle.write(key_json + "\t" + record_json + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def iter_sorted_chunk(path: Path) -> Iterable[tuple[HistoricalKey, dict[str, Any]]]:
    """Read a sorted run without materializing it in memory."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            key_text, record_text = line.rstrip("\n").split("\t", 1)
            raw_key = json.loads(key_text)
            if not isinstance(raw_key, list) or len(raw_key) != 3:
                raise ValueError(f"invalid historical sort key in {path}:{line_number}")
            key: HistoricalKey = (str(raw_key[0]), int(raw_key[1]), str(raw_key[2]))
            record = json.loads(record_text)
            if not isinstance(record, dict):
                raise ValueError(f"invalid historical record in {path}:{line_number}")
            yield key, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--existing-panel", type=Path, required=True)
    parser.add_argument("--stock-ids-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--historical-output", type=Path, default=None,
        help="Optional JSONL containing only appended historical rows with global row_index values",
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--sort-chunk-rows", type=int, default=2000,
        help="Maximum historical records held in memory per external-sort run",
    )
    parser.add_argument(
        "--sort-temp-dir", type=Path, default=None,
        help="Parent directory for sorted runs; use node-local scratch rather than Lustre",
    )
    args = parser.parse_args()

    if args.sort_chunk_rows < 1:
        raise ValueError("--sort-chunk-rows must be positive")
    sort_temp_parent = args.sort_temp_dir or Path(
        os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
    )
    sort_temp_parent.mkdir(parents=True, exist_ok=True)

    allowed_stocks = stock_ids_from_panel(args.stock_ids_from)
    existing = load_existing_panel(args.existing_panel)
    counters: dict[str, int] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.historical_output is not None:
        args.historical_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    historical_temporary = (
        args.historical_output.with_suffix(args.historical_output.suffix + ".tmp")
        if args.historical_output is not None else None
    )
    output_temporary.unlink(missing_ok=True)
    if historical_temporary is not None:
        historical_temporary.unlink(missing_ok=True)

    seen_ids: set[str] = set()
    current_rows = 0
    historical_rows = 0
    completed = False
    try:
        expected_stocks = existing["stock_id"].astype(str).str.strip().str.zfill(6).to_numpy()
        expected_dates = pd.to_datetime(existing["published_at"], errors="coerce", utc=True)

        # Stream current rows directly to the output. This preserves the frozen
        # 2018--2026 row-order contract without keeping 13 GiB of JSON in RAM.
        with output_temporary.open("w", encoding="utf-8") as output_handle:
            for record in iter_jsonl(args.current):
                if current_rows >= len(existing):
                    raise ValueError("current input contains more rows than the existing panel")
                item = normalize_frozen_record(record, period="2018_2026")
                if not text_value(item).strip():
                    counter_key = "2018_2026:empty_text_model_preserved"
                    counters[counter_key] = counters.get(counter_key, 0) + 1
                if stock_id(item) != expected_stocks[current_rows]:
                    raise ValueError(
                        "current stock order differs from existing panel at "
                        f"row_index={current_rows + 1}"
                    )
                panel_time = expected_dates.iloc[current_rows]
                record_time = publication_time(item)
                if pd.isna(panel_time) or record_time is None:
                    raise ValueError(f"invalid current publication date at row_index={current_rows + 1}")
                if panel_time.tz_convert(None).date() != record_time.date():
                    raise ValueError(
                        "current publication-date order differs at "
                        f"row_index={current_rows + 1}"
                    )
                item["row_index"] = current_rows + 1
                document_id = str(item["document_id"])
                if document_id in seen_ids:
                    raise ValueError(f"duplicate document_id after merge: {document_id}")
                seen_ids.add(document_id)
                output_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                current_rows += 1

            if current_rows != len(existing):
                raise ValueError(
                    "current input row count differs from the existing panel: "
                    f"current={current_rows} existing_panel={len(existing)}; refusing to remap embeddings"
                )

        # Historical records are externally sorted in small runs, then merged
        # by (stock, publication timestamp, document ID). Only a bounded run is
        # resident in RAM; the prior implementation held all historical JSON as
        # Python dictionaries and exceeded the 32 GiB Slurm limit.
        with tempfile.TemporaryDirectory(
            prefix="cninfo-unified-sort-", dir=str(sort_temp_parent)
        ) as temporary_directory:
            chunk_directory = Path(temporary_directory)
            chunk_paths: list[Path] = []
            chunk: list[tuple[HistoricalKey, dict[str, Any]]] = []
            chunk_index = 0
            for record in iter_jsonl(args.historical):
                item, reason = validate_record(record, allowed_stocks, period="2010_2017")
                if item is None:
                    assert reason is not None
                    counter_key = f"2010_2017:{reason}"
                    counters[counter_key] = counters.get(counter_key, 0) + 1
                    continue
                timestamp = publication_time(item)
                assert timestamp is not None
                if not 2010 <= timestamp.year <= 2017:
                    counters["2010_2017:outside_period"] = counters.get(
                        "2010_2017:outside_period", 0
                    ) + 1
                    continue
                document_id = str(item["document_id"])
                if document_id in seen_ids:
                    raise ValueError(f"duplicate document_id after merge: {document_id}")
                seen_ids.add(document_id)
                chunk.append((historical_sort_key(item), item))
                if len(chunk) >= args.sort_chunk_rows:
                    chunk_path = chunk_directory / f"run-{chunk_index:06d}.jsonl"
                    write_sorted_chunk(chunk_path, chunk)
                    chunk_paths.append(chunk_path)
                    chunk = []
                    chunk_index += 1
            if chunk:
                chunk_path = chunk_directory / f"run-{chunk_index:06d}.jsonl"
                write_sorted_chunk(chunk_path, chunk)
                chunk_paths.append(chunk_path)

            with contextlib.ExitStack() as stack:
                output_handle = stack.enter_context(output_temporary.open("a", encoding="utf-8"))
                historical_handle = (
                    stack.enter_context(historical_temporary.open("w", encoding="utf-8"))
                    if historical_temporary is not None else None
                )
                streams = [iter_sorted_chunk(path) for path in chunk_paths]
                next_row_index = len(existing) + 1
                for _key, item in heapq.merge(*streams, key=lambda value: value[0]):
                    item["row_index"] = next_row_index
                    line = json.dumps(item, ensure_ascii=False) + "\n"
                    output_handle.write(line)
                    if historical_handle is not None:
                        historical_handle.write(line)
                    historical_rows += 1
                    next_row_index += 1

        output_temporary.replace(args.output)
        if historical_temporary is not None:
            historical_temporary.replace(args.historical_output)
        summary = {
            "current_input": str(args.current),
            "historical_input": str(args.historical),
            "existing_panel": str(args.existing_panel),
            "stock_ids_from": str(args.stock_ids_from),
            "output": str(args.output),
            "historical_output": str(args.historical_output) if args.historical_output else None,
            "allowed_stocks": len(allowed_stocks),
            "current_rows": current_rows,
            "historical_rows": historical_rows,
            "rows": current_rows + historical_rows,
            "row_index_policy": "preserve current 1..N; append sorted historical rows",
            "document_id_unique": len(seen_ids) == current_rows + historical_rows,
            "rejected_records": counters,
            "historical_sort": "disk-backed bounded external merge sort",
            "sort_chunk_rows": args.sort_chunk_rows,
            "sort_temp_parent": str(sort_temp_parent),
        }
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        completed = True
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if not completed:
            output_temporary.unlink(missing_ok=True)
            if historical_temporary is not None:
                historical_temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
