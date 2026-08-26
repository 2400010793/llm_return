"""Audit Llama-2 causal readout shards, alignment, and contextual variation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


VARIANTS = ("short", "masked_short")
POOLING_NAMES = (
    "prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean",
    "cls", "title_max", "body_max", "title_body_max", "full_max",
)


def _metadata_indexes(
    path: Path, *, prompt_tokens: int,
) -> tuple[np.ndarray, list[str]]:
    values = []
    errors = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                values.append(row.get("row_index"))
                if row.get("sequence_layout") != "article_then_readout_prompt":
                    errors.append(f"metadata line {line_number}: invalid layout")
                start = row.get("prompt_start_zero_based")
                end = row.get("prompt_end_zero_based")
                saved = row.get("saved_token_count")
                if not all(isinstance(value, int) for value in (start, end, saved)):
                    errors.append(f"metadata line {line_number}: missing prompt positions")
                elif end - start + 1 != prompt_tokens or end != saved - 2:
                    errors.append(f"metadata line {line_number}: prompt is not intact at tail")
    return np.asarray(values), errors


def _sample_positions(rows: int, count: int = 16) -> np.ndarray:
    return np.unique(np.linspace(0, rows - 1, min(rows, count), dtype=np.int64))


def audit(args: argparse.Namespace) -> dict[str, object]:
    prompt = json.loads(
        args.prompt_spec.read_text(encoding="utf-8")
    )["prompts"]["short"]["text"]
    expected_hash = hashlib.sha256(str(prompt).encode()).hexdigest()
    errors: list[str] = []
    shard_rows: dict[str, int] = {}
    prompt_variation: list[float] = []
    mask_deltas: list[float] = []
    reference_ids: np.ndarray | None = None
    seen_short_indexes: set[int] = set()
    details = []
    for shard in range(args.expected_shards):
        variant_indexes: dict[str, np.ndarray] = {}
        variant_samples: dict[str, np.ndarray] = {}
        for variant in VARIANTS:
            model_directory = getattr(args, "model_directory", "llama2_13b")
            directory = args.root / f"shard-{shard}" / model_directory / variant
            required = (
                "COMPLETED", "summary.json", "metadata.jsonl", "short_pooling.npz",
                "prompt_token_embeddings.npy", "prompt_input_ids.npy",
                "prompt_tokens.json",
            )
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                errors.append(f"shard {shard} {variant}: missing {missing}")
                continue
            summary = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            rows = int(summary["rows"])
            shard_rows.setdefault(str(shard), rows)
            if shard_rows[str(shard)] != rows:
                errors.append(f"shard {shard}: variant row mismatch")
            expected_format = getattr(
                args, "expected_format", "llama2_causal_readout_v2",
            )
            if summary.get("format_version") != expected_format:
                errors.append(f"shard {shard} {variant}: obsolete format")
            if summary.get("sequence_layout") != "article_then_readout_prompt":
                errors.append(f"shard {shard} {variant}: invalid causal layout")
            if summary.get("prompt_sha256") != expected_hash:
                errors.append(f"shard {shard} {variant}: prompt hash mismatch")
            if summary.get("storage_dtype") != "float16":
                errors.append(f"shard {shard} {variant}: storage is not float16")
            if summary.get("cls_pooling") != "last_readout_token":
                errors.append(f"shard {shard} {variant}: cls is not causal readout")
            matrix = np.load(
                directory / "prompt_token_embeddings.npy", mmap_mode="r"
            )
            expected_shape = (
                rows,
                int(summary["prompt_token_count"]),
                int(summary["hidden_size"]),
            )
            if tuple(matrix.shape) != expected_shape:
                errors.append(
                    f"shard {shard} {variant}: prompt shape "
                    f"{matrix.shape} != {expected_shape}"
                )
            ids = np.load(directory / "prompt_input_ids.npy")
            if reference_ids is None:
                reference_ids = ids
            elif not np.array_equal(reference_ids, ids):
                errors.append(f"shard {shard} {variant}: prompt token ids differ")
            indexes, metadata_errors = _metadata_indexes(
                directory / "metadata.jsonl",
                prompt_tokens=int(summary["prompt_token_count"]),
            )
            errors.extend(
                f"shard {shard} {variant}: {message}"
                for message in metadata_errors
            )
            variant_indexes[variant] = indexes
            if len(indexes) != rows:
                errors.append(f"shard {shard} {variant}: metadata rows differ")
            if variant == "short":
                try:
                    keys = [int(value) for value in indexes]
                except (TypeError, ValueError):
                    errors.append(f"shard {shard}: non-integer row indexes")
                    keys = []
                duplicates = seen_short_indexes.intersection(keys)
                if duplicates or len(set(keys)) != len(keys):
                    errors.append(
                        f"shard {shard}: duplicate row indexes"
                    )
                seen_short_indexes.update(keys)
            positions = _sample_positions(rows)
            sample = np.asarray(matrix[positions], dtype=np.float32)
            variant_samples[variant] = sample
            if not np.isfinite(sample).all():
                errors.append(f"shard {shard} {variant}: non-finite prompt sample")
            flat = sample.reshape(len(sample), -1)
            variation = float(np.linalg.norm(flat - flat[:1], axis=1).max())
            prompt_variation.append(variation)
            with np.load(directory / "short_pooling.npz") as archive:
                if set(archive.files) != set(POOLING_NAMES):
                    errors.append(f"shard {shard} {variant}: pooling keys differ")
                for name in POOLING_NAMES:
                    if name not in archive:
                        continue
                    values = archive[name]
                    if values.shape != (rows, int(summary["hidden_size"])):
                        errors.append(
                            f"shard {shard} {variant}: {name} shape differs"
                        )
                    selected = np.asarray(values[positions], dtype=np.float32)
                    if not np.isfinite(selected).all():
                        errors.append(
                            f"shard {shard} {variant}: non-finite {name}"
                        )
            details.append({
                "shard": shard,
                "variant": variant,
                "rows": rows,
                "prompt_shape": list(matrix.shape),
                "prompt_sample_max_delta": variation,
            })
        if set(variant_indexes) == set(VARIANTS):
            if not np.array_equal(
                variant_indexes["short"], variant_indexes["masked_short"]
            ):
                errors.append(f"shard {shard}: masked/unmasked row indexes differ")
            delta = float(np.linalg.norm(
                (
                    variant_samples["short"] - variant_samples["masked_short"]
                ).reshape(len(variant_samples["short"]), -1),
                axis=1,
            ).mean())
            mask_deltas.append(delta)
    total_rows = sum(shard_rows.values())
    if len(shard_rows) != args.expected_shards:
        errors.append(
            f"expected {args.expected_shards} complete shards; "
            f"found {len(shard_rows)}"
        )
    if args.expected_rows is not None and total_rows != args.expected_rows:
        errors.append(f"expected {args.expected_rows} rows; found {total_rows}")
    if (
        args.expected_rows is not None
        and seen_short_indexes
        and not getattr(args, "allow_subset", False)
    ):
        invalid = [
            value for value in seen_short_indexes
            if value < 1 or value > args.expected_rows
        ]
        if invalid:
            errors.append(
                "row indexes do not exactly cover the expected 1-based panel"
            )
    maximum_variation = max(prompt_variation, default=0.0)
    mean_mask_delta = float(np.mean(mask_deltas)) if mask_deltas else 0.0
    if maximum_variation <= args.minimum_delta:
        errors.append("readout prompt embeddings do not vary across articles")
    if mean_mask_delta <= args.minimum_delta:
        errors.append("masked/unmasked readout embeddings do not differ")
    report = {
        "format_version": "llama2_causal_readout_audit_v1",
        "root": str(args.root.resolve()),
        "valid": not errors,
        "expected_shards": args.expected_shards,
        "shards_found": len(shard_rows),
        "rows": total_rows,
        "maximum_prompt_sample_delta": maximum_variation,
        "mean_mask_sample_delta": mean_mask_delta,
        "errors": errors,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    if args.strict and errors:
        raise ValueError("; ".join(errors))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompt-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--model-directory", default="llama2_13b")
    parser.add_argument(
        "--expected-format", default="llama2_causal_readout_v2",
    )
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="allow a preflight subset whose row indexes do not cover 1..expected_rows",
    )
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
