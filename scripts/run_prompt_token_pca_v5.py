"""Fit training-only sampled PCA to v5 flattened prompt tokens and transform all rows."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.token_gating import fit_token_gate


def shard_number(path: Path) -> int:
    return int(path.parents[1].name.split("-", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("data/processed/prompt_token_embeddings_v5"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--variant", choices=("short", "masked_short", "long", "masked_long"), required=True)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--fit-sample-rows", type=int, default=10000)
    parser.add_argument("--transform-batch-size", type=int, default=256)
    parser.add_argument("--token-gate-method", choices=("none", "fisher", "variance"), default="none")
    parser.add_argument("--token-gate-keep", type=int, default=8)
    parser.add_argument("--target-column", default="event_return_3d")
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--require-completed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    directories = sorted(args.input_root.glob(f"shard-*/{args.model}/{args.variant}"), key=shard_number)
    if len(directories) != args.expected_shards:
        raise ValueError(
            f"expected {args.expected_shards} input shards; found {len(directories)}"
        )
    shard_ids = [shard_number(directory) for directory in directories]
    if shard_ids != list(range(args.expected_shards)):
        raise ValueError(f"input shards are not contiguous: {shard_ids}")
    required = ["prompt_token_embeddings.npy", "metadata.jsonl"]
    if args.require_completed:
        required.extend(["summary.json", "COMPLETED"])
    for directory in directories:
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise ValueError(f"incomplete input shard {directory}: {missing}")
    panel = pd.read_parquet(
        args.panel, columns=["row_index", "entry_date", args.target_column],
    )
    if args.expected_rows is not None and len(panel) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} panel rows; found {len(panel)}")
    panel_rows = pd.to_numeric(panel["row_index"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(panel_rows), np.arange(1, len(panel) + 1)):
        raise ValueError("panel row_index must cover 1..N exactly once")
    panel["entry_date"] = pd.to_datetime(panel["entry_date"], errors="coerce")
    panel = panel.dropna(subset=["entry_date"])
    years = sorted(int(value) for value in panel["entry_date"].dt.year.unique())
    if len(years) < 9:
        raise ValueError(f"need at least 9 years; found {years}")
    pca_fit_years = years[:6]
    fit_rows = np.sort(
        panel.loc[panel["entry_date"].dt.year.isin(pca_fit_years), "row_index"].to_numpy(dtype=np.int64)
    )
    if len(fit_rows) < args.components:
        raise ValueError("not enough initial training rows for PCA")
    sample_size = min(args.fit_sample_rows, len(fit_rows))
    sample_positions = np.linspace(0, len(fit_rows) - 1, sample_size, dtype=np.int64)
    sampled_rows = set(fit_rows[sample_positions].tolist())
    target_lookup = panel.set_index("row_index")[args.target_column]

    samples: list[np.ndarray] = []
    sample_targets: list[np.ndarray] = []
    feature_dimension = None
    token_shape: tuple[int, int] | None = None
    metadata_by_directory: list[tuple[Path, np.ndarray]] = []
    for directory in directories:
        metadata = pd.read_json(directory / "metadata.jsonl", lines=True)
        row_indexes = metadata["row_index"].to_numpy(dtype=np.int64)
        matrix = np.load(directory / "prompt_token_embeddings.npy", mmap_mode="r")
        if matrix.ndim != 3:
            raise ValueError(f"expected [rows, tokens, hidden] in {directory}; found {matrix.shape}")
        current_token_shape = (int(matrix.shape[1]), int(matrix.shape[2]))
        if token_shape is None:
            token_shape = current_token_shape
        elif current_token_shape != token_shape:
            raise ValueError(f"token shape mismatch in {directory}: {current_token_shape} != {token_shape}")
        flat = matrix.reshape(len(matrix), -1)
        if feature_dimension is None:
            feature_dimension = int(flat.shape[1])
        elif flat.shape[1] != feature_dimension:
            raise ValueError(f"feature dimension mismatch in {directory}")
        local = np.flatnonzero(np.isin(row_indexes, list(sampled_rows)))
        if len(local):
            samples.append(np.asarray(flat[local], dtype=np.float32))
            sample_targets.append(
                pd.to_numeric(target_lookup.reindex(row_indexes[local]), errors="coerce").to_numpy(dtype=float)
            )
        metadata_by_directory.append((directory, row_indexes))
    sample = np.concatenate(samples, axis=0)
    if len(sample) != sample_size:
        raise ValueError(f"PCA sample coverage mismatch: {len(sample)} != {sample_size}")
    sampled_targets = np.concatenate(sample_targets)
    gate = None
    if args.token_gate_method != "none":
        if token_shape is None:
            raise ValueError("cannot fit token gate without token shape")
        gate = fit_token_gate(
            sample, sampled_targets, token_count=token_shape[0],
            keep_tokens=args.token_gate_keep, method=args.token_gate_method,
        )
        sample = gate.transform(sample)
    pca = PCA(
        n_components=args.components, svd_solver="randomized",
        iterated_power=3, n_oversamples=10, random_state=args.seed,
    )
    pca.fit(sample)
    del sample, samples

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate_tag = "" if gate is None else f"{gate.method}_gate{len(gate.selected_positions)}_"
    output_path = args.output_dir / f"{gate_tag}pca_{args.components}.npy"
    reduced = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(panel), args.components)
    )
    seen = np.zeros(len(panel), dtype=bool)
    for directory, row_indexes in metadata_by_directory:
        matrix = np.load(directory / "prompt_token_embeddings.npy", mmap_mode="r")
        flat = matrix.reshape(len(matrix), -1)
        for start in range(0, len(flat), args.transform_batch_size):
            stop = min(start + args.transform_batch_size, len(flat))
            positions = row_indexes[start:stop] - 1
            if (positions < 0).any() or (positions >= len(panel)).any() or seen[positions].any():
                raise ValueError(f"invalid or duplicate row_index in {directory}")
            transform_values = np.asarray(flat[start:stop], dtype=np.float32)
            if gate is not None:
                transform_values = gate.transform(transform_values)
            reduced[positions] = pca.transform(transform_values).astype(np.float32)
            seen[positions] = True
        reduced.flush()
    if not seen.all():
        raise ValueError(f"missing transformed rows: {int((~seen).sum())}")
    del reduced
    with (args.output_dir / f"{gate_tag}pca_{args.components}.pkl").open("wb") as handle:
        artifact = pca if gate is None else {"pca": pca, "token_gate": gate}
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    gate_summary = None
    if gate is not None:
        gate_summary = {
            "method": gate.method,
            "fit_scope": "same deterministic sample from initial six-year fit population as PCA",
            "keep_tokens": len(gate.selected_positions),
            "selected_positions_zero_based": gate.selected_positions.tolist(),
            "selected_positions_one_based": (gate.selected_positions + 1).tolist(),
            "selected_scores": gate.scores[gate.selected_positions].tolist(),
            "all_position_scores": gate.scores.tolist(),
            "hidden_size": gate.hidden_size,
            "gated_dimension": len(gate.selected_positions) * gate.hidden_size,
            "aggregation": "none; selected full token vectors flattened in original position order",
        }
    summary = {
        "format_version": "prompt_tokens_v5_sampled_training_only_gate_pca_v1",
        "model": args.model, "variant": args.variant,
        "input_shards": len(directories), "rows": len(panel),
        "input_dimension": feature_dimension, "components": args.components,
        "token_shape": list(token_shape) if token_shape is not None else None,
        "token_gate": gate_summary,
        "target_column": args.target_column,
        "preprocessing_protocol": (
            "PCA and token gate fit once on the earliest six-year fit population, "
            "then frozen for every leakage-safe rolling classifier fold"
        ),
        "pca_fit_years": pca_fit_years, "pca_fit_population_rows": len(fit_rows),
        "pca_fit_sample_rows": sample_size,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "output": str(output_path), "row_order": "source row_index ascending, position=row_index-1",
    }
    summary_path = (
        args.output_dir / "summary.json" if gate is None
        else args.output_dir / f"{gate_tag}pca_{args.components}.summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
