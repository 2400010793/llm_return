"""Rank every frozen v5 prompt-token position using training rows only.

This script reads existing token embeddings in batches and never runs a
Transformer. It reports separate 2018--2023 fit and 2018--2025 all-train gates
so final-model token selection does not reuse the validation-period gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.artifacts import atomic_json, file_fingerprint
from src.models.token_gating import TokenGate, fit_streaming_token_gate


def shard_number(path: Path) -> int:
    return int(path.parents[1].name.split("-", 1)[1])


def gate_report(gate: TokenGate, tokens: list[str], years: list[int]) -> dict[str, object]:
    order = np.lexsort((np.arange(gate.token_count), -gate.scores))
    ranking = [
        {
            "rank": rank,
            "position_zero_based": int(position),
            "position_one_based": int(position + 1),
            "token": tokens[position],
            "score": float(gate.scores[position]),
            "selected": bool(position in gate.selected_positions),
        }
        for rank, position in enumerate(order, start=1)
    ]
    return {
        "years": years,
        "method": gate.method,
        "token_count": gate.token_count,
        "hidden_size": gate.hidden_size,
        "keep_tokens": len(gate.selected_positions),
        "selected_positions_zero_based": gate.selected_positions.tolist(),
        "selected_positions_one_based": (gate.selected_positions + 1).tolist(),
        "all_position_scores": gate.scores.tolist(),
        "ranking": ranking,
    }


def token_chunks(
    directories: list[Path],
    lookup: pd.DataFrame,
    *,
    years: set[int],
    batch_size: int,
    expected_shape: tuple[int, int],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for directory in directories:
        metadata = pd.read_json(directory / "metadata.jsonl", lines=True)
        if "row_index" not in metadata:
            raise ValueError(f"metadata lacks row_index: {directory}")
        row_indexes = pd.to_numeric(metadata["row_index"], errors="raise").to_numpy(dtype=np.int64)
        if len(np.unique(row_indexes)) != len(row_indexes):
            raise ValueError(f"duplicate row_index in {directory}")
        aligned = lookup.reindex(row_indexes)
        if aligned["year"].isna().any():
            raise ValueError(f"panel does not cover metadata rows in {directory}")
        matrix = np.load(directory / "prompt_token_embeddings.npy", mmap_mode="r")
        if matrix.ndim != 3 or tuple(matrix.shape[1:]) != expected_shape:
            raise ValueError(f"token shape mismatch in {directory}: {matrix.shape}")
        if len(matrix) != len(row_indexes):
            raise ValueError(f"metadata/matrix row mismatch in {directory}")
        eligible = aligned["year"].astype(int).isin(years).to_numpy()
        targets = aligned["target"].to_numpy(dtype=float)
        for start in range(0, len(matrix), batch_size):
            stop = min(start + batch_size, len(matrix))
            mask = eligible[start:stop]
            if not mask.any():
                continue
            values = np.asarray(matrix[start:stop][mask], dtype=np.float32)
            yield values.reshape(len(values), -1), targets[start:stop][mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("data/processed/prompt_token_embeddings_v5"),
    )
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument(
        "--variant", choices=("short", "masked_short", "long", "masked_long"),
        required=True,
    )
    parser.add_argument("--target-column", default="next_day_return")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--row-index-column", default="row_index")
    parser.add_argument("--method", choices=("fisher", "variance"), default="fisher")
    parser.add_argument("--keep-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    directories = sorted(
        args.input_root.glob(f"shard-*/{args.model}/{args.variant}"), key=shard_number,
    )
    if args.expected_shards and len(directories) != args.expected_shards:
        raise ValueError(f"expected {args.expected_shards} input shards; found {len(directories)}")
    if not directories:
        raise ValueError("no prompt-token embedding shards found")
    required_files = ("metadata.jsonl", "prompt_token_embeddings.npy", "prompt_tokens.json")
    for directory in directories:
        missing = [name for name in required_files if not (directory / name).is_file()]
        if missing:
            raise ValueError(f"incomplete prompt-token shard {directory}: {missing}")

    prompt = json.loads((directories[0] / "prompt_tokens.json").read_text(encoding="utf-8"))
    tokens = [str(token) for token in prompt["tokens"]]
    for directory in directories[1:]:
        candidate = json.loads((directory / "prompt_tokens.json").read_text(encoding="utf-8"))
        if candidate.get("tokens") != prompt["tokens"]:
            raise ValueError(f"prompt tokens differ in {directory}")
    first_matrix = np.load(directories[0] / "prompt_token_embeddings.npy", mmap_mode="r")
    expected_shape = (int(first_matrix.shape[1]), int(first_matrix.shape[2]))
    if len(tokens) != expected_shape[0]:
        raise ValueError("prompt token list and matrix token dimension differ")

    spec = {
        "format_version": "prompt_token_global_ranking_v1",
        "panel": file_fingerprint(args.panel, hash_content=True),
        "model": args.model, "variant": args.variant,
        "target_column": args.target_column, "date_column": args.date_column,
        "row_index_column": args.row_index_column, "method": args.method,
        "keep_tokens": args.keep_tokens, "batch_size": args.batch_size,
        "token_shape": list(expected_shape), "prompt_text": prompt.get("text"),
        "prompt_tokens": tokens,
        "shards": [
            {
                "directory": str(directory),
                "metadata": file_fingerprint(directory / "metadata.jsonl", hash_content=True),
                "matrix": file_fingerprint(directory / "prompt_token_embeddings.npy"),
            }
            for directory in directories
        ],
    }
    experiment_id = hashlib.sha256(
        json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if args.output.is_file() and not args.force_recompute:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("experiment_id") == experiment_id:
            print(json.dumps({"output": str(args.output), "experiment_id": experiment_id, "resumed": True}))
            return

    panel = pd.read_parquet(
        args.panel,
        columns=[args.row_index_column, args.date_column, args.target_column],
    )
    if panel[args.row_index_column].duplicated().any():
        raise ValueError("panel row_index must be unique")
    dates = pd.to_datetime(panel[args.date_column], errors="coerce")
    lookup = pd.DataFrame({
        "year": dates.dt.year.to_numpy(),
        "target": pd.to_numeric(panel[args.target_column], errors="coerce").to_numpy(),
    }, index=pd.to_numeric(panel[args.row_index_column], errors="raise").to_numpy(dtype=np.int64))
    years = sorted(int(year) for year in lookup["year"].dropna().unique())
    if len(years) < 9:
        raise ValueError(f"need at least 9 years; found {years}")
    fit_years, all_train_years = years[:6], years[:8]
    fit_gate = fit_streaming_token_gate(
        token_chunks(
            directories, lookup, years=set(fit_years), batch_size=args.batch_size,
            expected_shape=expected_shape,
        ),
        token_count=expected_shape[0], keep_tokens=args.keep_tokens, method=args.method,
    )
    all_train_gate = fit_streaming_token_gate(
        token_chunks(
            directories, lookup, years=set(all_train_years), batch_size=args.batch_size,
            expected_shape=expected_shape,
        ),
        token_count=expected_shape[0], keep_tokens=args.keep_tokens, method=args.method,
    )
    report = {
        "experiment_id": experiment_id,
        "spec": spec,
        "fit": gate_report(fit_gate, tokens, fit_years),
        "all_train": gate_report(all_train_gate, tokens, all_train_years),
        "interpretation": (
            "Global contextualized prompt-position relevance; not per-document attribution, "
            "attention, SHAP, or a causal token effect."
        ),
    }
    atomic_json(args.output, report)
    print(json.dumps({
        "output": str(args.output), "experiment_id": experiment_id,
        "fit_selected_tokens": [tokens[position] for position in fit_gate.selected_positions],
        "all_train_selected_tokens": [tokens[position] for position in all_train_gate.selected_positions],
        "resumed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
