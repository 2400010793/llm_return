"""Audit the cleaned 2010--2026 CNINFO prompt-token embedding store."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED = {
    "roberta": (28, 768),
    "bge_m3": (18, 1024),
}
VARIANTS = ("short", "masked_short")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate_directory(
    directory: Path,
    *,
    model: str,
    variant: str,
    sample_values: bool = True,
) -> tuple[dict[str, Any], int]:
    required = (
        "prompt_token_embeddings.npy",
        "prompt_input_ids.npy",
        "prompt_tokens.json",
        "metadata.jsonl",
        "summary.json",
        "COMPLETED",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"missing {missing} in {directory}")

    summary = load_json(directory / "summary.json")
    if summary.get("format_version") != "prompt_tokens_v5_bos_offset_fixed":
        raise ValueError(f"unexpected format_version in {directory}")
    if summary.get("model") != model or summary.get("variant") != variant:
        raise ValueError(f"identity mismatch in {directory}")
    if not summary.get("input") or not summary.get("input_parts"):
        raise ValueError(f"input provenance is missing in {directory}")

    token_count, hidden_size = EXPECTED[model]
    matrix = np.load(directory / "prompt_token_embeddings.npy", mmap_mode="r")
    expected_shape = (int(summary["rows"]), token_count, hidden_size)
    if matrix.shape != expected_shape or matrix.dtype != np.float32:
        raise ValueError(
            f"matrix mismatch in {directory}: {matrix.shape}/{matrix.dtype} "
            f"!= {expected_shape}/float32"
        )
    if summary.get("shape") != list(expected_shape):
        raise ValueError(f"declared shape mismatch in {directory}")
    if summary.get("prompt_slice") != f"[1:{1 + token_count}] after BOS":
        raise ValueError(f"BOS offset declaration mismatch in {directory}")

    input_ids = np.load(directory / "prompt_input_ids.npy", mmap_mode="r")
    if input_ids.shape != (token_count,) or not np.issubdtype(input_ids.dtype, np.integer):
        raise ValueError(f"prompt input IDs mismatch in {directory}")
    tokens = load_json(directory / "prompt_tokens.json")
    if len(tokens.get("tokens", [])) != token_count:
        raise ValueError(f"prompt token text count mismatch in {directory}")
    if sample_values and len(matrix):
        indices = sorted({0, len(matrix) // 2, len(matrix) - 1})
        if not all(np.isfinite(np.asarray(matrix[index])).all() for index in indices):
            raise ValueError(f"non-finite sampled embedding in {directory}")
    return summary, int(summary["rows"])


def audit(args: argparse.Namespace) -> dict[str, Any]:
    expected_rows = int(args.rows)
    models = tuple(args.models)
    variants = tuple(args.variants)
    seen = {
        (model, variant): np.zeros(expected_rows + 1, dtype=np.bool_)
        for model in models
        for variant in variants
    }
    prompt_ids: dict[str, np.ndarray] = {}
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for shard in range(args.num_shards):
        for model in models:
            for variant in variants:
                directory = args.root / f"shard-{shard}" / model / variant
                try:
                    summary, declared_rows = validate_directory(
                        directory, model=model, variant=variant,
                    )
                    ids = np.asarray(
                        np.load(directory / "prompt_input_ids.npy", mmap_mode="r")
                    )
                    prior = prompt_ids.setdefault(model, ids.copy())
                    if not np.array_equal(prior, ids):
                        raise ValueError(f"prompt token IDs changed in {directory}")

                    metadata_rows = 0
                    with (directory / "metadata.jsonl").open(encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, start=1):
                            if not line.strip():
                                continue
                            item = json.loads(line)
                            if int(item.get("output_row", -1)) != metadata_rows:
                                raise ValueError(
                                    f"output_row mismatch at {directory}:{line_number}"
                                )
                            row_index = int(item["row_index"])
                            if not 1 <= row_index <= expected_rows:
                                raise ValueError(
                                    f"row_index out of range in {directory}: {row_index}"
                                )
                            key = (model, variant)
                            if seen[key][row_index]:
                                raise ValueError(
                                    f"duplicate row_index for {model}/{variant}: {row_index}"
                                )
                            seen[key][row_index] = True
                            if item.get("prompt_offset_start") != 1:
                                raise ValueError(f"wrong prompt start in {directory}")
                            if item.get("prompt_offset_end_exclusive") != 1 + EXPECTED[model][0]:
                                raise ValueError(f"wrong prompt end in {directory}")
                            metadata_rows += 1
                    if metadata_rows != declared_rows:
                        raise ValueError(
                            f"metadata rows mismatch in {directory}: "
                            f"{metadata_rows} != {declared_rows}"
                        )
                    results.append({
                        "shard": shard,
                        "model": model,
                        "variant": variant,
                        "rows": declared_rows,
                        "shape": summary["shape"],
                        "input": summary["input"],
                    })
                except Exception as exc:  # collect every failed shard in one report
                    errors.append(f"{directory}: {exc}")

    coverage: list[dict[str, Any]] = []
    for key, flags in seen.items():
        count = int(flags[1:].sum())
        missing = np.flatnonzero(~flags[1:])[:20] + 1
        if count != expected_rows:
            errors.append(
                f"{key[0]}/{key[1]} coverage {count} != {expected_rows}; "
                f"missing examples={missing.tolist()}"
            )
        coverage.append({
            "model": key[0],
            "variant": key[1],
            "rows": count,
            "expected_rows": expected_rows,
            "complete": count == expected_rows,
        })

    return {
        "status": "passed" if not errors else "failed",
        "root": str(args.root),
        "expected_rows": expected_rows,
        "num_shards": args.num_shards,
        "directories_checked": len(results),
        "models": list(models),
        "variants": list(variants),
        "expected_directories": args.num_shards * len(models) * len(variants),
        "coverage": coverage,
        "errors": errors,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/processed/prompt_token_embeddings_2010_2026_clean_v1"),
    )
    parser.add_argument("--rows", type=int, default=903665)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(EXPECTED), default=list(EXPECTED),
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/cninfo_prompt_token_embeddings_2010_2026_clean_v1_audit.json"),
    )
    args = parser.parse_args()
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": report["status"],
        "directories_checked": report["directories_checked"],
        "expected_directories": report["expected_directories"],
        "coverage": report["coverage"],
        "errors": report["errors"][:20],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
