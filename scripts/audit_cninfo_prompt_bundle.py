"""Audit CNINFO prompt/mask inputs against the frozen v3/v4 construction."""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_cninfo_prompt_bundle import (
    PROMPT_VERSION,
    TEMPLATE_HASHES,
    build_prompt_record,
)
from scripts.build_cninfo_prompt_inputs import (
    COMPANY_NAME,
    DATE_PATTERNS,
    SUBJECT_MASK,
    TIME_MASK,
)


SHARED_FIELDS = (
    "row_index",
    "document_id",
    "stock_id_alignment",
    "announcement_date_alignment",
    "text_plain",
    "text_input_2_short",
    "text_input_3_long",
    "text_input_5_masked_short",
    "text_input_6_masked_long",
    "text_plain_sha256",
    "text_input_2_short_sha256",
    "text_input_3_long_sha256",
    "text_input_5_masked_short_sha256",
    "text_input_6_masked_long_sha256",
    "short_prompt",
    "short_title",
    "short_body",
    "masked_short_prompt",
    "masked_short_title",
    "masked_short_body",
)
SHORT_DOWNSTREAM_FIELDS = SHARED_FIELDS
LONG_DOWNSTREAM_FIELDS = SHARED_FIELDS + (
    "text_input_7_fixed_long",
    "text_input_8_fixed_masked_long",
    "long_prompt",
    "long_title",
    "long_body",
    "masked_long_prompt",
    "masked_long_title",
    "masked_long_body",
)


def _read_shard(root: Path, shard: int) -> Iterator[dict[str, Any]]:
    files = sorted((root / f"shard-{shard}").glob("part-*.jsonl"))
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"record is not an object: {path}:{line_number}")
                yield value


def _merged_records(root: Path, num_shards: int) -> Iterator[dict[str, Any]]:
    iterators = [_read_shard(root, shard) for shard in range(num_shards)]
    heap: list[tuple[int, int, dict[str, Any], Iterator[dict[str, Any]]]] = []
    for shard, iterator in enumerate(iterators):
        try:
            record = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(record.get("row_index", -1)), shard, record, iterator))
    while heap:
        _, shard, record, iterator = heapq.heappop(heap)
        yield record
        try:
            next_record = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (int(next_record.get("row_index", -1)), shard, next_record, iterator),
        )


def _check_manifest(
    path: Path,
    source_rows: int,
    num_shards: int,
    *,
    expected_rows: int | None = None,
    expected_shard: int | None = None,
    row_offset: int = 0,
) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(f"unexpected prompt version in {path}")
    if int(manifest.get("source_rows", -1)) != source_rows:
        raise ValueError(f"manifest source row mismatch in {path}")
    if int(manifest.get("row_index_start", row_offset + 1)) != row_offset + 1:
        raise ValueError(f"manifest row start mismatch in {path}")
    if int(manifest.get("row_index_end", row_offset + source_rows)) != row_offset + source_rows:
        raise ValueError(f"manifest row end mismatch in {path}")
    rows = int(manifest.get("rows", -1))
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"manifest output row mismatch in {path}: {rows} != {expected_rows}")
    if rows < 0 or rows > source_rows:
        raise ValueError(f"manifest output row count out of range in {path}: {rows}")
    if int(manifest.get("num_shards", -1)) != num_shards:
        raise ValueError(f"manifest shard mismatch in {path}")
    if expected_shard is not None and int(manifest.get("shard_id", -1)) != expected_shard:
        raise ValueError(f"manifest shard id mismatch in {path}")
    if "parts" in manifest:
        part_rows = sum(int(part.get("rows", -1)) for part in manifest["parts"])
    elif "shards" in manifest:
        part_rows = sum(int(shard.get("rows", -1)) for shard in manifest["shards"])
    else:
        raise ValueError(f"manifest has neither parts nor shards: {path}")
    if part_rows != rows:
        raise ValueError(f"manifest child row mismatch in {path}")
    return rows


def _residual_leakage(record: dict[str, Any]) -> dict[str, bool]:
    """Check the same identity/time leakage classes used by the old audit."""
    name = str(record.get("stock_name", "") or "")
    stock_id = str(record.get("stock_id", "") or "")
    title = str(record.get("title_clean_final", record.get("title", "")) or "")
    text = str(record.get("text_model", record.get("text", "")) or "")
    masked_title = str(record.get("masked_title", "") or "")
    masked_body = str(record.get("masked_body", "") or "")
    masked_fields = (masked_title, masked_body)
    normalized_masked_fields = tuple(
        re.sub(r"\s+", "", value) for value in masked_fields
    )
    normalized_name = re.sub(r"\s+", "", name)
    return {
        "stock_name_remaining": bool(
            len(normalized_name) >= 2
            and normalized_name != SUBJECT_MASK
            and any(
                normalized_name in value for value in normalized_masked_fields
            )
        ),
        "stock_code_remaining": bool(
            stock_id
            and any(
                re.search(rf"(?<!\d){re.escape(stock_id)}(?!\d)", value)
                for value in masked_fields
            )
        ),
        "legal_company_name_remaining": any(
            COMPANY_NAME.search(value) for value in masked_fields
        ),
        "calendar_date_remaining": any(
            pattern.search(value)
            for value in masked_fields
            for pattern in DATE_PATTERNS
        ),
        "title_was_masked": masked_title != title,
        "body_was_masked": masked_body != text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--short-dir", type=Path, required=True)
    parser.add_argument("--long-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--row-offset", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("num-shards must be positive")

    input_rows = 0
    checked_rows = 0
    mismatches: list[dict[str, Any]] = []
    mismatch_counts: Counter[str] = Counter()
    alignment_counts: Counter[str] = Counter()
    downstream_missing_counts: Counter[str] = Counter()
    residual_counts: Counter[str] = Counter()
    mask_counts: Counter[str] = Counter()
    changed_counts: Counter[str] = Counter()
    short_duplicate_rows = 0
    long_duplicate_rows = 0
    short_gap_rows = 0
    long_gap_rows = 0
    short_out_of_order_rows = 0
    long_out_of_order_rows = 0
    previous_short_index: int | None = None
    previous_long_index: int | None = None
    residual_examples: list[dict[str, Any]] = []
    short_records = _merged_records(args.short_dir, args.num_shards)
    long_records = _merged_records(args.long_dir, args.num_shards)

    with args.input.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            input_rows += 1
            source_record = json.loads(line)
            expected_row_index = input_rows + args.row_offset
            try:
                short = next(short_records)
                long = next(long_records)
            except StopIteration as exc:
                raise ValueError(f"prompt bundle ended before input row {input_rows}") from exc
            row_errors: list[str] = []
            expected_short = build_prompt_record(source_record, expected_row_index, layout="short")
            expected_long = build_prompt_record(source_record, expected_row_index, layout="long")
            for label, actual, expected in (
                ("short", short, expected_short),
                ("long", long, expected_long),
            ):
                actual_keys = set(actual)
                expected_keys = set(expected)
                for field in sorted(expected_keys - actual_keys):
                    key = f"{label}.missing.{field}"
                    row_errors.append(key)
                    mismatch_counts[key] += 1
                for field in sorted(actual_keys - expected_keys):
                    key = f"{label}.extra.{field}"
                    row_errors.append(key)
                    mismatch_counts[key] += 1
                for field, expected_value in expected.items():
                    if actual.get(field) != expected_value:
                        key = f"{label}.{field}"
                        row_errors.append(key)
                        mismatch_counts[key] += 1
            for field in sorted(set(expected_short) & set(expected_long)):
                if short.get(field) != long.get(field):
                    key = f"short_long.{field}"
                    row_errors.append(key)
                    mismatch_counts[key] += 1
            if expected_short["short_prompt"] != expected_short["masked_short_prompt"]:
                row_errors.append("short_prompt_pair")
                mismatch_counts["short_prompt_pair"] += 1
            if expected_long["long_prompt"] != expected_long["masked_long_prompt"]:
                row_errors.append("long_prompt_pair")
                mismatch_counts["long_prompt_pair"] += 1

            for field in ("document_id", "stock_id_alignment", "announcement_date_alignment"):
                if short.get(field) == expected_short.get(field) == long.get(field):
                    alignment_counts[f"{field}_matched"] += 1
                else:
                    alignment_counts[f"{field}_mismatched"] += 1

            for field in SHORT_DOWNSTREAM_FIELDS:
                if field not in short:
                    downstream_missing_counts[field] += 1
            for field in LONG_DOWNSTREAM_FIELDS:
                if field not in long:
                    downstream_missing_counts[f"long.{field}"] += 1

            for label, actual in (("short", short), ("long", long)):
                for field in ("masked_short_title", "masked_short_body"):
                    mask_counts[f"{label}.{field}_subject"] += str(actual.get(field, "") or "").count(SUBJECT_MASK)
                    mask_counts[f"{label}.{field}_time"] += str(actual.get(field, "") or "").count(TIME_MASK)
                if label == "long":
                    for field in ("masked_long_title", "masked_long_body"):
                        mask_counts[f"{label}.{field}_subject"] += str(actual.get(field, "") or "").count(SUBJECT_MASK)
                        mask_counts[f"{label}.{field}_time"] += str(actual.get(field, "") or "").count(TIME_MASK)

            for label, actual in (("short", short), ("long", long)):
                leakage_input = dict(source_record)
                prefix = "masked_long" if label == "long" else "masked_short"
                leakage_input["masked_title"] = actual.get(f"{prefix}_title")
                leakage_input["masked_body"] = actual.get(f"{prefix}_body")
                leakage = _residual_leakage(leakage_input)
                for check, failed in leakage.items():
                    if check.endswith("_was_masked"):
                        changed_counts[f"{label}.{check}"] += int(failed)
                    elif failed:
                        residual_counts[f"{label}.{check}"] += 1
                        if len(residual_examples) < 20:
                            residual_examples.append({
                                "row_index": input_rows,
                                "bundle": label,
                                "check": check,
                            })

            for actual, previous, bundle in (
                (short, previous_short_index, "short"),
                (long, previous_long_index, "long"),
            ):
                try:
                    current = int(actual.get("row_index", -1))
                except (TypeError, ValueError):
                    current = -1
                if previous is not None:
                    if current == previous:
                        if bundle == "short":
                            short_duplicate_rows += 1
                        else:
                            long_duplicate_rows += 1
                    elif current > previous + 1:
                        if bundle == "short":
                            short_gap_rows += current - previous - 1
                        else:
                            long_gap_rows += current - previous - 1
                    elif current <= previous:
                        if bundle == "short":
                            short_out_of_order_rows += 1
                        else:
                            long_out_of_order_rows += 1
                if bundle == "short":
                    previous_short_index = current
                else:
                    previous_long_index = current
            if row_errors and len(mismatches) < 20:
                mismatches.append({"row_index": expected_row_index, "fields": row_errors[:20]})
            checked_rows += 1

    extra_short_count = sum(1 for _ in short_records)
    extra_long_count = sum(1 for _ in long_records)
    extra_short = extra_short_count > 0
    extra_long = extra_long_count > 0

    bundle_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if bundle_manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("unexpected prompt version in bundle manifest")
    if int(bundle_manifest.get("source_rows", -1)) != checked_rows:
        raise ValueError("bundle manifest source row mismatch")
    if bundle_manifest.get("template_hashes") != TEMPLATE_HASHES:
        raise ValueError("bundle manifest template hash mismatch")
    short_root_rows = _check_manifest(
        args.short_dir / "manifest.json", checked_rows, args.num_shards,
        row_offset=args.row_offset,
    )
    long_root_rows = _check_manifest(
        args.long_dir / "manifest.json", checked_rows, args.num_shards,
        row_offset=args.row_offset,
    )
    if short_root_rows != long_root_rows:
        raise ValueError("short and long root manifest row counts differ")
    short_manifest_rows = 0
    long_manifest_rows = 0
    for shard in range(args.num_shards):
        short_manifest_rows += _check_manifest(
            args.short_dir / f"shard-{shard}.manifest.json",
            checked_rows,
            args.num_shards,
            expected_rows=None,
            expected_shard=shard,
            row_offset=args.row_offset,
        )
        long_manifest_rows += _check_manifest(
            args.long_dir / f"shard-{shard}.manifest.json",
            checked_rows,
            args.num_shards,
            expected_rows=None,
            expected_shard=shard,
            row_offset=args.row_offset,
        )
    if short_manifest_rows != checked_rows or long_manifest_rows != checked_rows:
        raise ValueError("shard manifest rows do not cover the input exactly")

    duplicate_or_gap = any((
        short_duplicate_rows, long_duplicate_rows,
        short_gap_rows, long_gap_rows,
        short_out_of_order_rows, long_out_of_order_rows,
        extra_short_count, extra_long_count,
    ))
    alignment_failed = any(key.endswith("_mismatched") for key in alignment_counts)
    downstream_failed = bool(downstream_missing_counts)
    residual_failed = bool(residual_counts)
    summary = {
        "status": "passed" if not (
            mismatches or duplicate_or_gap or alignment_failed or downstream_failed or residual_failed
        ) else "failed",
        "prompt_version": PROMPT_VERSION,
        "input": str(args.input),
        "short_dir": str(args.short_dir),
        "long_dir": str(args.long_dir),
        "input_rows": input_rows,
        "checked_rows": checked_rows,
        "expected_row_index": [args.row_offset + 1, args.row_offset + checked_rows],
        "short_extra_rows": extra_short,
        "long_extra_rows": extra_long,
        "short_extra_row_count": extra_short_count,
        "long_extra_row_count": extra_long_count,
        "row_index_audit": {
            "short_duplicate_rows": short_duplicate_rows,
            "long_duplicate_rows": long_duplicate_rows,
            "short_gap_rows": short_gap_rows,
            "long_gap_rows": long_gap_rows,
            "short_out_of_order_rows": short_out_of_order_rows,
            "long_out_of_order_rows": long_out_of_order_rows,
        },
        "alignment_audit": dict(alignment_counts),
        "template_hashes": TEMPLATE_HASHES,
        "masking_audit": {
            "subject_placeholder": SUBJECT_MASK,
            "time_placeholder": TIME_MASK,
            "replacement_counts": dict(mask_counts),
            "changed_title_or_body_counts": dict(changed_counts),
            "residual_leakage_counts": dict(residual_counts),
            "residual_leakage_examples": residual_examples,
        },
        "downstream_compatibility": {
            "short_required_fields": list(SHORT_DOWNSTREAM_FIELDS),
            "long_required_fields": list(LONG_DOWNSTREAM_FIELDS),
            "missing_field_counts": dict(downstream_missing_counts),
        },
        "mismatch_count_sampled": len(mismatches),
        "mismatch_counts": dict(mismatch_counts),
        "mismatches": mismatches,
        "consistency_checks": {
            "prompt_and_mask_pairing": True,
            "all_expected_fields_equal_frozen_builder": not mismatches,
            "short_long_bundles_identical": not mismatches,
            "row_coverage_exact": not duplicate_or_gap,
            "stock_date_document_alignment_exact": not alignment_failed,
            "mask_residual_leakage_absent": not residual_failed,
            "downstream_fields_present": not downstream_failed,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
