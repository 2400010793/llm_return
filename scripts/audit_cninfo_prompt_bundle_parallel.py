"""Run the CNINFO prompt-bundle audit with one worker per shard.

The original audit merges all 32 short and long shard streams in one Python
process. This module keeps the same per-record checks, but first partitions
the historical input once and then audits the 32 independent shard streams in
parallel. The reducer combines the shard summaries and performs the global
manifest checks without rereading the 166 GiB prompt bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_cninfo_prompt_bundle import (  # noqa: E402
    LONG_DOWNSTREAM_FIELDS,
    SHORT_DOWNSTREAM_FIELDS,
    _check_manifest,
    _read_shard,
    _residual_leakage,
)
from scripts.build_cninfo_prompt_bundle import (  # noqa: E402
    PROMPT_VERSION,
    TEMPLATE_HASHES,
    build_prompt_record,
)
from scripts.build_cninfo_prompt_inputs import (  # noqa: E402
    SUBJECT_MASK,
    TIME_MASK,
)


COUNTER_NAMES = (
    "mismatch_counts",
    "alignment_counts",
    "downstream_missing_counts",
    "residual_counts",
    "mask_counts",
    "changed_counts",
)


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def split_source(
    input_path: Path,
    output_dir: Path,
    manifest_path: Path,
    *,
    num_shards: int,
    row_offset: int,
) -> None:
    """Partition the input once so workers do not reread the 43 GiB source."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("shard-*.jsonl"):
        path.unlink()
    manifest_path.unlink(missing_ok=True)

    temporary_paths = [
        output_dir / f".shard-{shard}.jsonl.tmp.{os.getpid()}"
        for shard in range(num_shards)
    ]
    handles = []
    counts = [0] * num_shards
    source_rows = 0
    try:
        handles = [path.open("w", encoding="utf-8") for path in temporary_paths]
        with input_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                shard = source_rows % num_shards
                handles[shard].write(line if line.endswith("\n") else line + "\n")
                counts[shard] += 1
                source_rows += 1
    finally:
        for handle in handles:
            handle.close()

    for temporary, shard in zip(temporary_paths, range(num_shards)):
        os.replace(temporary, output_dir / f"shard-{shard}.jsonl")

    manifest = {
        "input": str(input_path),
        "source_rows": source_rows,
        "row_index_start": row_offset + 1,
        "row_index_end": row_offset + source_rows,
        "num_shards": num_shards,
        "shards": [
            {
                "shard_id": shard,
                "path": str(output_dir / f"shard-{shard}.jsonl"),
                "rows": counts[shard],
            }
            for shard in range(num_shards)
        ],
    }
    _write_atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False))


def _empty_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {name: Counter() for name in COUNTER_NAMES}
    metrics.update(
        {
            "mismatches": [],
            "residual_examples": [],
            "short_duplicate_rows": 0,
            "long_duplicate_rows": 0,
            "short_gap_rows": 0,
            "long_gap_rows": 0,
            "short_out_of_order_rows": 0,
            "long_out_of_order_rows": 0,
        }
    )
    return metrics


def _audit_pair(
    source_record: dict[str, Any],
    expected_row_index: int,
    short: dict[str, Any],
    long: dict[str, Any],
) -> dict[str, Any]:
    metrics = _empty_metrics()
    row_errors: list[str] = []
    expected_short = build_prompt_record(
        source_record, expected_row_index, layout="short"
    )
    expected_long = build_prompt_record(
        source_record, expected_row_index, layout="long"
    )
    for label, actual, expected in (
        ("short", short, expected_short),
        ("long", long, expected_long),
    ):
        actual_keys = set(actual)
        expected_keys = set(expected)
        for field in sorted(expected_keys - actual_keys):
            key = f"{label}.missing.{field}"
            row_errors.append(key)
            metrics["mismatch_counts"][key] += 1
        for field in sorted(actual_keys - expected_keys):
            key = f"{label}.extra.{field}"
            row_errors.append(key)
            metrics["mismatch_counts"][key] += 1
        for field, expected_value in expected.items():
            if actual.get(field) != expected_value:
                key = f"{label}.{field}"
                row_errors.append(key)
                metrics["mismatch_counts"][key] += 1

    for field in sorted(set(expected_short) & set(expected_long)):
        if short.get(field) != long.get(field):
            key = f"short_long.{field}"
            row_errors.append(key)
            metrics["mismatch_counts"][key] += 1
    if expected_short["short_prompt"] != expected_short["masked_short_prompt"]:
        row_errors.append("short_prompt_pair")
        metrics["mismatch_counts"]["short_prompt_pair"] += 1
    if expected_long["long_prompt"] != expected_long["masked_long_prompt"]:
        row_errors.append("long_prompt_pair")
        metrics["mismatch_counts"]["long_prompt_pair"] += 1

    for field in ("document_id", "stock_id_alignment", "announcement_date_alignment"):
        if short.get(field) == expected_short.get(field) == long.get(field):
            metrics["alignment_counts"][f"{field}_matched"] += 1
        else:
            metrics["alignment_counts"][f"{field}_mismatched"] += 1

    for field in SHORT_DOWNSTREAM_FIELDS:
        if field not in short:
            metrics["downstream_missing_counts"][field] += 1
    for field in LONG_DOWNSTREAM_FIELDS:
        if field not in long:
            metrics["downstream_missing_counts"][f"long.{field}"] += 1

    for label, actual in (("short", short), ("long", long)):
        for field in ("masked_short_title", "masked_short_body"):
            metrics["mask_counts"][f"{label}.{field}_subject"] += str(
                actual.get(field, "") or ""
            ).count(SUBJECT_MASK)
            metrics["mask_counts"][f"{label}.{field}_time"] += str(
                actual.get(field, "") or ""
            ).count(TIME_MASK)
        if label == "long":
            for field in ("masked_long_title", "masked_long_body"):
                metrics["mask_counts"][f"{label}.{field}_subject"] += str(
                    actual.get(field, "") or ""
                ).count(SUBJECT_MASK)
                metrics["mask_counts"][f"{label}.{field}_time"] += str(
                    actual.get(field, "") or ""
                ).count(TIME_MASK)

    for label, actual in (("short", short), ("long", long)):
        leakage_input = dict(source_record)
        prefix = "masked_long" if label == "long" else "masked_short"
        leakage_input["masked_title"] = actual.get(f"{prefix}_title")
        leakage_input["masked_body"] = actual.get(f"{prefix}_body")
        leakage = _residual_leakage(leakage_input)
        for check, failed in leakage.items():
            if check.endswith("_was_masked"):
                metrics["changed_counts"][f"{label}.{check}"] += int(failed)
            elif failed:
                metrics["residual_counts"][f"{label}.{check}"] += 1
                if len(metrics["residual_examples"]) < 20:
                    metrics["residual_examples"].append(
                        {
                            "row_index": expected_row_index,
                            "bundle": label,
                            "check": check,
                        }
                    )

    if row_errors:
        metrics["mismatches"].append(
            {"row_index": expected_row_index, "fields": row_errors[:20]}
        )
    return metrics


def _merge_metrics(target: dict[str, Any], source: dict[str, Any]) -> None:
    for name in COUNTER_NAMES:
        target[name].update(source.get(name, {}))
    for name in (
        "short_duplicate_rows",
        "long_duplicate_rows",
        "short_gap_rows",
        "long_gap_rows",
        "short_out_of_order_rows",
        "long_out_of_order_rows",
    ):
        target[name] += int(source.get(name, 0))
    target["mismatches"].extend(source.get("mismatches", []))
    target["mismatches"] = target["mismatches"][:20]
    target["residual_examples"].extend(source.get("residual_examples", []))
    target["residual_examples"] = target["residual_examples"][:20]


def _json_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    for name in COUNTER_NAMES:
        result[name] = dict(metrics[name])
    return result


def audit_shard(
    source_dir: Path,
    short_dir: Path,
    long_dir: Path,
    output_path: Path,
    *,
    shard: int,
    num_shards: int,
    row_offset: int,
) -> None:
    if not 0 <= shard < num_shards:
        raise ValueError(f"invalid shard {shard} for {num_shards} shards")
    source_path = source_dir / f"shard-{shard}.jsonl"
    metrics = _empty_metrics()
    short_records = _read_shard(short_dir, shard)
    long_records = _read_shard(long_dir, shard)
    checked_rows = 0
    previous_short: int | None = None
    previous_long: int | None = None

    with source_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            checked_rows += 1
            source_position = shard + 1 + (checked_rows - 1) * num_shards
            expected_row_index = row_offset + source_position
            source_record = json.loads(line)
            if not isinstance(source_record, dict):
                raise ValueError(f"source record is not an object: {source_path}")
            try:
                short = next(short_records)
                long = next(long_records)
            except StopIteration as exc:
                raise ValueError(
                    f"prompt bundle ended before source row {source_position}"
                ) from exc

            pair_metrics = _audit_pair(
                source_record, expected_row_index, short, long
            )
            _merge_metrics(metrics, pair_metrics)
            for actual, previous, bundle in (
                (short, previous_short, "short"),
                (long, previous_long, "long"),
            ):
                try:
                    current = int(actual.get("row_index", -1))
                except (TypeError, ValueError):
                    current = -1
                if previous is not None:
                    if current == previous:
                        metrics[f"{bundle}_duplicate_rows"] += 1
                    elif current > previous + num_shards:
                        metrics[f"{bundle}_gap_rows"] += current - previous - num_shards
                    elif current <= previous:
                        metrics[f"{bundle}_out_of_order_rows"] += 1
                if bundle == "short":
                    previous_short = current
                else:
                    previous_long = current

    extra_short_count = sum(1 for _ in short_records)
    extra_long_count = sum(1 for _ in long_records)
    duplicate_or_gap = any(
        (
            metrics["short_duplicate_rows"],
            metrics["long_duplicate_rows"],
            metrics["short_gap_rows"],
            metrics["long_gap_rows"],
            metrics["short_out_of_order_rows"],
            metrics["long_out_of_order_rows"],
            extra_short_count,
            extra_long_count,
        )
    )
    failed = bool(
        metrics["mismatches"]
        or duplicate_or_gap
        or any(key.endswith("_mismatched") for key in metrics["alignment_counts"])
        or metrics["downstream_missing_counts"]
        or metrics["residual_counts"]
    )
    summary = _json_metrics(metrics)
    summary.update(
        {
            "status": "failed" if failed else "passed",
            "shard_id": shard,
            "num_shards": num_shards,
            "input_rows": checked_rows,
            "checked_rows": checked_rows,
            "first_row_index": row_offset + shard + 1 if checked_rows else None,
            "last_row_index": (
                row_offset + shard + 1 + (checked_rows - 1) * num_shards
                if checked_rows
                else None
            ),
            "short_extra_row_count": extra_short_count,
            "long_extra_row_count": extra_long_count,
        }
    )
    _write_atomic_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


def _check_bundle_manifests(
    bundle_manifest_path: Path,
    short_dir: Path,
    long_dir: Path,
    *,
    source_rows: int,
    num_shards: int,
    row_offset: int,
) -> None:
    bundle = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    if bundle.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("unexpected prompt version in bundle manifest")
    if int(bundle.get("source_rows", -1)) != source_rows:
        raise ValueError("bundle manifest source row mismatch")
    if bundle.get("template_hashes") != TEMPLATE_HASHES:
        raise ValueError("bundle manifest template hash mismatch")
    if int(bundle.get("row_index_start", -1)) != row_offset + 1:
        raise ValueError("bundle manifest row start mismatch")
    if int(bundle.get("row_index_end", -1)) != row_offset + source_rows:
        raise ValueError("bundle manifest row end mismatch")

    short_rows = _check_manifest(
        short_dir / "manifest.json", source_rows, num_shards, row_offset=row_offset
    )
    long_rows = _check_manifest(
        long_dir / "manifest.json", source_rows, num_shards, row_offset=row_offset
    )
    if short_rows != source_rows or long_rows != source_rows:
        raise ValueError("root prompt manifests do not cover the input")
    for shard in range(num_shards):
        _check_manifest(
            short_dir / f"shard-{shard}.manifest.json",
            source_rows,
            num_shards,
            expected_shard=shard,
            row_offset=row_offset,
        )
        _check_manifest(
            long_dir / f"shard-{shard}.manifest.json",
            source_rows,
            num_shards,
            expected_shard=shard,
            row_offset=row_offset,
        )


def reduce_audit(
    source_manifest_path: Path,
    bundle_manifest_path: Path,
    short_dir: Path,
    long_dir: Path,
    shard_summary_dir: Path,
    output_path: Path,
    *,
    num_shards: int,
    row_offset: int,
) -> None:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_rows = int(source_manifest.get("source_rows", -1))
    if source_rows < 0 or int(source_manifest.get("num_shards", -1)) != num_shards:
        raise ValueError("invalid source split manifest")
    summaries = []
    for shard in range(num_shards):
        path = shard_summary_dir / f"shard-{shard}.json"
        if not path.is_file():
            raise ValueError(f"missing shard audit summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if int(summary.get("shard_id", -1)) != shard:
            raise ValueError(f"wrong shard ID in {path}")
        summaries.append(summary)

    _check_bundle_manifests(
        bundle_manifest_path,
        short_dir,
        long_dir,
        source_rows=source_rows,
        num_shards=num_shards,
        row_offset=row_offset,
    )

    metrics = _empty_metrics()
    for summary in summaries:
        _merge_metrics(metrics, summary)
    checked_rows = sum(int(summary.get("checked_rows", -1)) for summary in summaries)
    expected_shard_rows = [
        (source_rows - shard + num_shards - 1) // num_shards
        if source_rows > shard
        else 0
        for shard in range(num_shards)
    ]
    actual_shard_rows = [int(summary.get("checked_rows", -1)) for summary in summaries]
    if checked_rows != source_rows or actual_shard_rows != expected_shard_rows:
        raise ValueError(
            f"source shard coverage mismatch: actual={actual_shard_rows} "
            f"expected={expected_shard_rows}"
        )

    duplicate_or_gap = any(
        (
            metrics["short_duplicate_rows"],
            metrics["long_duplicate_rows"],
            metrics["short_gap_rows"],
            metrics["long_gap_rows"],
            metrics["short_out_of_order_rows"],
            metrics["long_out_of_order_rows"],
            sum(int(summary.get("short_extra_row_count", 0)) for summary in summaries),
            sum(int(summary.get("long_extra_row_count", 0)) for summary in summaries),
        )
    )
    alignment_failed = any(
        key.endswith("_mismatched") for key in metrics["alignment_counts"]
    )
    downstream_failed = bool(metrics["downstream_missing_counts"])
    residual_failed = bool(metrics["residual_counts"])
    worker_failed = any(summary.get("status") != "passed" for summary in summaries)
    extra_short_count = sum(
        int(summary.get("short_extra_row_count", 0)) for summary in summaries
    )
    extra_long_count = sum(
        int(summary.get("long_extra_row_count", 0)) for summary in summaries
    )
    failed = bool(
        worker_failed
        or metrics["mismatches"]
        or duplicate_or_gap
        or alignment_failed
        or downstream_failed
        or residual_failed
    )
    summary = _json_metrics(metrics)
    summary.update(
        {
            "status": "failed" if failed else "passed",
            "prompt_version": PROMPT_VERSION,
            "input": source_manifest.get("input"),
            "source_split_manifest": str(source_manifest_path),
            "short_dir": str(short_dir),
            "long_dir": str(long_dir),
            "input_rows": source_rows,
            "checked_rows": checked_rows,
            "expected_row_index": [row_offset + 1, row_offset + source_rows],
            "short_extra_rows": extra_short_count > 0,
            "long_extra_rows": extra_long_count > 0,
            "short_extra_row_count": extra_short_count,
            "long_extra_row_count": extra_long_count,
            "row_index_audit": {
                "short_duplicate_rows": metrics["short_duplicate_rows"],
                "long_duplicate_rows": metrics["long_duplicate_rows"],
                "short_gap_rows": metrics["short_gap_rows"],
                "long_gap_rows": metrics["long_gap_rows"],
                "short_out_of_order_rows": metrics["short_out_of_order_rows"],
                "long_out_of_order_rows": metrics["long_out_of_order_rows"],
            },
            "template_hashes": TEMPLATE_HASHES,
            "parallel_workers": num_shards,
            "worker_summaries": [
                {
                    "shard_id": int(item["shard_id"]),
                    "status": item.get("status"),
                    "checked_rows": int(item.get("checked_rows", -1)),
                }
                for item in summaries
            ],
            "masking_audit": {
                "subject_placeholder": SUBJECT_MASK,
                "time_placeholder": TIME_MASK,
                "replacement_counts": dict(metrics["mask_counts"]),
                "changed_title_or_body_counts": dict(metrics["changed_counts"]),
                "residual_leakage_counts": dict(metrics["residual_counts"]),
                "residual_leakage_examples": metrics["residual_examples"],
            },
            "downstream_compatibility": {
                "short_required_fields": list(SHORT_DOWNSTREAM_FIELDS),
                "long_required_fields": list(LONG_DOWNSTREAM_FIELDS),
                "missing_field_counts": dict(metrics["downstream_missing_counts"]),
            },
            "mismatch_count_sampled": len(metrics["mismatches"]),
            "mismatch_counts": dict(metrics["mismatch_counts"]),
            "mismatches": metrics["mismatches"],
            "consistency_checks": {
                "prompt_and_mask_pairing": True,
                "all_expected_fields_equal_frozen_builder": not metrics["mismatches"],
                "short_long_bundles_identical": not metrics["mismatches"],
                "row_coverage_exact": not duplicate_or_gap and not worker_failed,
                "stock_date_document_alignment_exact": not alignment_failed,
                "mask_residual_leakage_absent": not residual_failed,
                "downstream_fields_present": not downstream_failed,
            },
        }
    )
    _write_atomic_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--input", type=Path, required=True)
    split_parser.add_argument("--output-dir", type=Path, required=True)
    split_parser.add_argument("--manifest", type=Path, required=True)
    split_parser.add_argument("--num-shards", type=int, default=32)
    split_parser.add_argument("--row-offset", type=int, default=0)

    shard_parser = subparsers.add_parser("shard")
    shard_parser.add_argument("--source-dir", type=Path, required=True)
    shard_parser.add_argument("--short-dir", type=Path, required=True)
    shard_parser.add_argument("--long-dir", type=Path, required=True)
    shard_parser.add_argument("--output", type=Path, required=True)
    shard_parser.add_argument("--shard", type=int, required=True)
    shard_parser.add_argument("--num-shards", type=int, default=32)
    shard_parser.add_argument("--row-offset", type=int, default=0)

    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--source-manifest", type=Path, required=True)
    reduce_parser.add_argument("--bundle-manifest", type=Path, required=True)
    reduce_parser.add_argument("--short-dir", type=Path, required=True)
    reduce_parser.add_argument("--long-dir", type=Path, required=True)
    reduce_parser.add_argument("--shard-summary-dir", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    reduce_parser.add_argument("--num-shards", type=int, default=32)
    reduce_parser.add_argument("--row-offset", type=int, default=0)

    args = parser.parse_args()
    if args.command == "split":
        split_source(
            args.input,
            args.output_dir,
            args.manifest,
            num_shards=args.num_shards,
            row_offset=args.row_offset,
        )
    elif args.command == "shard":
        audit_shard(
            args.source_dir,
            args.short_dir,
            args.long_dir,
            args.output,
            shard=args.shard,
            num_shards=args.num_shards,
            row_offset=args.row_offset,
        )
    else:
        reduce_audit(
            args.source_manifest,
            args.bundle_manifest,
            args.short_dir,
            args.long_dir,
            args.shard_summary_dir,
            args.output,
            num_shards=args.num_shards,
            row_offset=args.row_offset,
        )


if __name__ == "__main__":
    main()