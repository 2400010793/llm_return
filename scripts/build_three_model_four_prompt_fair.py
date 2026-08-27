#!/usr/bin/env python3
"""Build a strict common panel and token/body matrices for three models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROMPTS = {
    "profit": "盈利",
    "return": "收益",
    "excess_return": "超额收益",
    "loss": "亏损",
}
MODELS = ("roberta", "bge_m3", "qwen3_embedding_8b")


def json_default(value):
    """Serialize numpy scalars produced by pandas/numpy manifests."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    frame["row_index"] = pd.to_numeric(frame["row_index"], errors="raise").astype(np.int64)
    if frame["row_index"].duplicated().any():
        raise ValueError(f"duplicate row_index: {path}")
    return frame


def normalized(token: str) -> str:
    return token.replace("▁", "").replace("Ġ", "").strip()


def target_indices(tokens: list[str], phrase: str) -> list[int]:
    clean = [normalized(token) for token in tokens]
    matches: list[list[int]] = []
    for start in range(len(clean)):
        joined = ""
        for stop in range(start, len(clean)):
            joined += clean[stop]
            if joined == phrase:
                matches.append(list(range(start, stop + 1)))
            if len(joined) >= len(phrase):
                break
    if not matches:
        raise ValueError(f"cannot locate target phrase {phrase!r} in {tokens}")
    return matches[-1]


def write_matrix(path: Path, arrays: list[np.ndarray], order: np.ndarray) -> None:
    rows = sum(len(array) for array in arrays)
    dimensions = {array.shape[1] for array in arrays}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent dimensions: {dimensions}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    result = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(rows, dimensions.pop()),
    )
    cursor = 0
    for array in arrays:
        stop = cursor + len(array)
        result[cursor:stop] = np.asarray(array, dtype=np.float32)
        cursor = stop
    result.flush()
    del result
    if not np.array_equal(order, np.arange(rows)):
        source = np.load(temporary, mmap_mode="r")
        sorted_path = temporary.with_suffix(".sorted.npy")
        sorted_result = np.lib.format.open_memmap(
            sorted_path, mode="w+", dtype=np.float32, shape=source.shape,
        )
        for start in range(0, rows, 1024):
            selected = order[start:start + 1024]
            sorted_result[start:start + len(selected)] = source[selected]
        sorted_result.flush()
        del sorted_result, source
        temporary.unlink()
        sorted_path.replace(path)
    else:
        temporary.replace(path)


def common_qwen_rows(qwen_root: Path) -> tuple[list[int], list[int]]:
    shard_sets = []
    for prompt in PROMPTS:
        base = qwen_root / prompt / "masked_short"
        shard_sets.append({
            int(path.name.split("-")[1]) for path in base.glob("shard-*")
            if (path / "COMPLETED").is_file()
        })
    common_shards = sorted(set.intersection(*shard_sets))
    row_sets = []
    for prompt in PROMPTS:
        rows: set[int] = set()
        for shard in common_shards:
            leaf = qwen_root / prompt / "masked_short" / f"shard-{shard}"
            rows.update(read_metadata(leaf / "metadata.jsonl")["row_index"].tolist())
        row_sets.append(rows)
    common_rows = sorted(set.intersection(*row_sets))
    return common_shards, common_rows


def collect_qwen(
    qwen_root: Path, prompt: str, phrase: str, shards: list[int], selected_raw: set[int],
    row_index_offset: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    frames, token_arrays, body_arrays = [], [], []
    token_specs = []
    for shard in shards:
        leaf = qwen_root / prompt / "masked_short" / f"shard-{shard}"
        metadata = read_metadata(leaf / "metadata.jsonl")
        keep = metadata["row_index"].isin(selected_raw).to_numpy()
        if not keep.any():
            continue
        spec = json.loads((leaf / "prompt_tokens.json").read_text(encoding="utf-8"))
        indices = target_indices(list(spec["tokens"]), phrase)
        token_specs.append((tuple(spec["tokens"]), tuple(indices), spec["text"]))
        token_values = np.load(leaf / "prompt_token_embeddings.npy", mmap_mode="r")
        body_values = np.load(leaf / "article_mean_embeddings.npy", mmap_mode="r")
        selected_metadata = metadata.loc[keep, ["row_index"]].copy()
        selected_metadata["row_index"] += row_index_offset
        frames.append(selected_metadata)
        token_arrays.append(np.asarray(token_values[keep][:, indices, :], dtype=np.float32).mean(axis=1))
        body_arrays.append(np.asarray(body_values[keep], dtype=np.float32))
    if len(set(token_specs)) != 1:
        raise ValueError(f"Qwen token spec changed across shards: {prompt}")
    frame = pd.concat(frames, ignore_index=True)
    return frame, np.concatenate(token_arrays), np.concatenate(body_arrays), {
        "tokens": list(token_specs[0][0]), "target_indices": list(token_specs[0][1]),
        "prompt_text": token_specs[0][2],
    }


def collect_encoder(
    embedding_root: Path, model: str, prompt: str, phrase: str, selected_raw: set[int],
    encoder_shards: int, row_index_offset: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    frames, token_arrays, body_arrays = [], [], []
    token_specs = []
    for shard in range(encoder_shards):
        shard_root = embedding_root / model / f"shard-{shard}"
        leaf = shard_root / prompt / "masked_short"
        if not (shard_root / "COMPLETED").is_file():
            raise FileNotFoundError(shard_root / "COMPLETED")
        for required in (
            "metadata.jsonl", "prompt_tokens.json",
            "prompt_token_embeddings.npy", "short_pooling.npz",
        ):
            if not (leaf / required).is_file():
                raise FileNotFoundError(leaf / required)
        metadata = read_metadata(leaf / "metadata.jsonl")
        keep = metadata["row_index"].isin(selected_raw).to_numpy()
        if not keep.any():
            continue
        spec = json.loads((leaf / "prompt_tokens.json").read_text(encoding="utf-8"))
        indices = target_indices(list(spec["tokens"]), phrase)
        token_specs.append((tuple(spec["tokens"]), tuple(indices), spec["text"]))
        token_values = np.load(leaf / "prompt_token_embeddings.npy", mmap_mode="r")
        pooling = np.load(leaf / "short_pooling.npz", mmap_mode="r")
        selected_metadata = metadata.loc[keep, ["row_index"]].copy()
        selected_metadata["row_index"] += row_index_offset
        frames.append(selected_metadata)
        token_arrays.append(np.asarray(token_values[keep][:, indices, :], dtype=np.float32).mean(axis=1))
        body_arrays.append(np.asarray(pooling["body_mean"][keep], dtype=np.float32))
    if len(set(token_specs)) != 1:
        raise ValueError(f"encoder token spec changed across shards: {model}/{prompt}")
    frame = pd.concat(frames, ignore_index=True)
    return frame, np.concatenate(token_arrays), np.concatenate(body_arrays), {
        "tokens": list(token_specs[0][0]), "target_indices": list(token_specs[0][1]),
        "prompt_text": token_specs[0][2],
    }


def distribution(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    result["entry_date"] = pd.to_datetime(result["entry_date"], errors="coerce").dt.normalize()
    result["year"] = result["entry_date"].dt.year
    return result.groupby("year", as_index=False).agg(
        rows=("row_index", "size"),
        stocks=("stock_id", "nunique"),
        stock_days=("stock_id", lambda _: 0),
        finite_returns=("next_day_return", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        return_mean=("next_day_return", "mean"),
        return_std=("next_day_return", "std"),
    ).assign(sample=label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("sina", "cninfo"), required=True)
    parser.add_argument("--encoder-shards", type=int, required=True)
    parser.add_argument("--row-index-offset", type=int, default=0)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--history-years", type=int, required=True)
    parser.add_argument("--validation-years", type=int, required=True)
    args = parser.parse_args()

    shards, row_indices = common_qwen_rows(args.qwen_root)
    if args.expected_rows is not None and len(row_indices) != args.expected_rows:
        raise ValueError(f"expected frozen {args.expected_rows:,}-row intersection, got {len(row_indices):,}")
    selected_raw = set(row_indices)
    canonical_rows = [row + args.row_index_offset for row in row_indices]
    selected = set(canonical_rows)
    panel = pd.read_parquet(args.panel)
    if panel["row_index"].duplicated().any():
        raise ValueError("canonical panel has duplicate row_index")
    subset = panel[panel["row_index"].isin(selected)].copy().sort_values("row_index")
    if len(subset) != len(selected) or subset["row_index"].tolist() != canonical_rows:
        raise ValueError("canonical panel does not cover the frozen row_index intersection")
    subset["entry_date"] = pd.to_datetime(subset["entry_date"], errors="raise").dt.normalize()
    output = args.output_root
    (output / "intersection").mkdir(parents=True, exist_ok=True)
    panel_output = output / "intersection" / "panel.parquet"
    # The frozen intersection is immutable. Reuse a valid existing panel so
    # reruns do not rewrite hundreds of MB on Lustre unnecessarily.
    if not panel_output.is_file():
        # Rolling/geometry stages only need these scalar columns. Avoid
        # serializing nested object columns (for example stock-id arrays)
        # which are unnecessarily large and fragile on the shared filesystem.
        panel_columns = [
            column for column in (
                "row_index", "document_id", "stock_id", "stock_name",
                "entry_date", "next_day_return", "next_day_open_to_open_return",
            ) if column in subset.columns
        ]
        subset[panel_columns].to_parquet(panel_output, index=False)

    full_dist = distribution(panel, "full")
    selected_dist = distribution(subset, "intersection")
    audit = pd.concat([full_dist, selected_dist], ignore_index=True)
    stock_days = (
        subset.groupby(subset.entry_date.dt.year)
        .apply(lambda x: len(x[["stock_id", "entry_date"]].drop_duplicates()), include_groups=False)
    )
    audit.loc[audit["sample"].eq("intersection"), "stock_days"] = audit.loc[
        audit["sample"].eq("intersection"), "year"
    ].map(stock_days).to_numpy()
    full_panel = panel.copy()
    full_panel["entry_date"] = pd.to_datetime(full_panel["entry_date"], errors="coerce").dt.normalize()
    full_stock_days = full_panel.groupby(full_panel.entry_date.dt.year).apply(
        lambda x: len(x[["stock_id", "entry_date"]].drop_duplicates()), include_groups=False,
    )
    audit.loc[audit["sample"].eq("full"), "stock_days"] = audit.loc[
        audit["sample"].eq("full"), "year"
    ].map(full_stock_days).to_numpy()
    audit.to_csv(output / "intersection" / "year_distribution.csv", index=False)

    config_rows = []
    for model in MODELS:
        for prompt, phrase in PROMPTS.items():
            if model == "qwen3_embedding_8b":
                frame, token, body, token_spec = collect_qwen(
                    args.qwen_root, prompt, phrase, shards, selected_raw, args.row_index_offset,
                )
            else:
                frame, token, body, token_spec = collect_encoder(
                    args.encoder_root, model, prompt, phrase, selected_raw,
                    args.encoder_shards, args.row_index_offset,
                )
            order = np.argsort(frame["row_index"].to_numpy(), kind="stable")
            ordered_rows = frame.iloc[order]["row_index"].to_numpy()
            if not np.array_equal(ordered_rows, subset["row_index"].to_numpy()):
                raise ValueError(f"row mismatch: {model}/{prompt}")
            for representation, matrix in (("token", token), ("body", body)):
                destination = output / "matrices" / model / prompt / representation
                matrix_path = destination / "matrix.npy"
                write_matrix(matrix_path, [matrix], order)
                pd.DataFrame({"row_index": ordered_rows}).to_parquet(
                    destination / "metadata.parquet", index=False,
                )
                matrix_hash = sha256_file(matrix_path)
                manifest = {
                    "format_version": "three_model_four_prompt_matrix_v1",
                    "model": model, "prompt": prompt, "target_phrase": phrase,
                    "variant": "masked_short", "representation": representation,
                    "pooling": "target_span_mean" if representation == "token" else (
                        "article_mean" if model == "qwen3_embedding_8b" else "body_mean"
                    ),
                    "rows": len(matrix), "dimension": matrix.shape[1],
                    "prompt_text": token_spec["prompt_text"],
                    "prompt_tokens": token_spec["tokens"],
                    "target_indices": token_spec["target_indices"],
                    "sha256": matrix_hash,
                }
                (destination / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
                config_rows.append({
                    "config_id": len(config_rows), "model": model, "prompt": prompt,
                    "target_phrase": phrase, "representation": representation,
                    "matrix": str(matrix_path),
                    "metadata": str(destination / "metadata.parquet"),
                })

    config = pd.DataFrame(config_rows)
    config.to_csv(output / "intersection" / "config_manifest.tsv", sep="\t", index=False)
    # Keep the frozen row intersection intact, but do not let a small number
    # of rows with missing dates create a NaN-to-int failure or a fake fold.
    available_years = {
        int(year) for year in subset.loc[subset.entry_date.notna(), "entry_date"].dt.year.unique()
    }
    test_years = [
        year for year in sorted(available_years)
        if set(range(year - args.history_years, year + 1)).issubset(available_years)
    ]
    if not test_years:
        raise ValueError(
            f"{args.dataset} has no complete {args.history_years}+1 calendar window"
        )
    fold_rows = []
    for row in config.itertuples(index=False):
        for test_year in test_years:
            fold_rows.append({
                "task_id": len(fold_rows), "config_id": row.config_id,
                "model": row.model, "prompt": row.prompt,
                "target_phrase": row.target_phrase,
                "representation": row.representation,
                "matrix": row.matrix, "metadata": row.metadata,
                "test_year": test_year, "history_years": args.history_years,
            })
    pd.DataFrame(fold_rows).to_csv(
        output / "intersection" / "fold_manifest.tsv", sep="\t", index=False,
    )
    root_manifest = {
        "format_version": "three_model_four_prompt_fair_pca_v2",
        "dataset": args.dataset,
        "variant": "masked_short", "rows": len(subset),
        "stock_days": len(subset[["stock_id", "entry_date"]].drop_duplicates()),
        "date_min": str(subset.entry_date.min().date()),
        "date_max": str(subset.entry_date.max().date()),
        "qwen_common_shards": shards,
        "encoder_shards": args.encoder_shards,
        "row_index_offset": args.row_index_offset,
        "raw_row_index_min": min(row_indices), "raw_row_index_max": max(row_indices),
        "canonical_row_index_min": min(canonical_rows), "canonical_row_index_max": max(canonical_rows),
        "test_years": test_years, "folds": len(fold_rows),
        "history_years": args.history_years,
        "validation_years_within_history": args.validation_years,
        "rolling_protocol": (
            f"{args.history_years - args.validation_years}+{args.validation_years}+1"
            if args.dataset == "sina" else f"{args.history_years}+1"
        ),
        "fixed_parameters": {
            "pca_components": 32, "ridge_alpha": 100.0,
            "kmeans_k": 6, "kmeans_seeds": [17, 29, 42, 71, 113],
            "soft_shrinkage": 500.0, "soft_temperature": 0.5,
            "umap_components": 8, "hdbscan_min_cluster_size": 50,
            "hdbscan_min_samples": 10,
        },
        "models": list(MODELS), "prompts": PROMPTS,
        "qwen_period_limitation": True,
        "body_pooling": {"roberta": "body_mean", "bge_m3": "body_mean", "qwen3_embedding_8b": "article_mean"},
    }
    (output / "intersection" / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8",
    )
    print(json.dumps(root_manifest, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
