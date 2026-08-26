"""Build compact, row-aligned mechanism caches from sharded prompt tensors."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import MODEL_PATHS
from src.analysis.prompt_token_mechanisms import (
    TARGET_COLUMNS,
    YearlyTokenMoments,
    aggregate_moments,
    rolling_windows,
    token_metrics_from_moments,
    validate_tokenizer_mapping,
)
from src.data.prompt_token_embeddings import PromptTokenEmbeddingStore
from src.evaluation.artifacts import atomic_json


MODELS = ("roberta", "bge_m3", "ckip_bert", "xlm_roberta_large")
PROMPT_LENGTHS = ("short", "long")


def _lookups(panel: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    row_indexes = pd.to_numeric(panel["row_index"], errors="raise").to_numpy(dtype=np.int64)
    if len(np.unique(row_indexes)) != len(row_indexes) or (row_indexes < 1).any():
        raise ValueError("panel row_index must be unique and positive")
    size = int(row_indexes.max()) + 1
    output_position = np.full(size, -1, dtype=np.int64)
    output_position[row_indexes] = np.arange(len(panel), dtype=np.int64)
    dates = pd.to_datetime(panel["entry_date"], errors="coerce")
    years = np.full(size, -1, dtype=np.int32)
    valid_dates = dates.notna().to_numpy()
    years[row_indexes[valid_dates]] = dates.loc[valid_dates].dt.year.to_numpy(dtype=np.int32)
    targets = {}
    for target in TARGET_COLUMNS:
        values = np.full(size, np.nan, dtype=np.float64)
        values[row_indexes] = pd.to_numeric(panel[target], errors="coerce").to_numpy(dtype=float)
        targets[target] = values
    return years, targets, output_position


def _group_mean(values: np.ndarray, positions: list[np.ndarray]) -> np.ndarray:
    return np.stack(
        [values[:, selected, :].mean(axis=1, dtype=np.float32) for selected in positions],
        axis=1,
    ).astype(np.float32, copy=False)


def _fisher_topk(
    moments_path: Path, windows: list[dict[str, object]], keep: int,
) -> tuple[np.ndarray, dict[str, dict[str, list[int]]]]:
    with np.load(moments_path) as archive:
        targets = []
        report: dict[str, dict[str, list[int]]] = {}
        for target in TARGET_COLUMNS:
            target_rows = []
            report[target] = {}
            for window in windows:
                test_year = int(window["test_year"])
                moments = aggregate_moments(
                    archive, window["fit_years"], target=target,
                )
                metrics = token_metrics_from_moments(moments)
                order = np.lexsort((np.arange(len(metrics["fisher"])), -metrics["fisher"]))
                selected = np.sort(order[:keep]).astype(np.int64)
                target_rows.append(selected)
                report[target][str(test_year)] = selected.tolist()
            targets.append(np.stack(target_rows, axis=0))
    return np.stack(targets, axis=0), report


def build_cache(args: argparse.Namespace) -> dict[str, object]:
    variant = args.prompt_length
    masked_variant = f"masked_{variant}"
    base = PromptTokenEmbeddingStore(
        args.embedding_root, model=args.model, variant=variant,
        expected_shards=args.expected_shards, expected_rows=args.expected_rows,
    )
    masked = PromptTokenEmbeddingStore(
        args.embedding_root, model=args.model, variant=masked_variant,
        expected_shards=args.expected_shards, expected_rows=args.expected_rows,
    )
    if not np.array_equal(base.row_indexes, masked.row_indexes):
        raise ValueError("masked and unmasked stores contain different row_index sets")
    if base.prompt_tokens != masked.prompt_tokens or base.token_count != masked.token_count:
        raise ValueError("masked and unmasked prompt token definitions differ")

    panel = pd.read_parquet(
        args.panel, columns=["row_index", "entry_date", *TARGET_COLUMNS, "stock_id"],
    )
    panel = panel.sort_values("row_index", kind="stable").reset_index(drop=True)
    if len(panel) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} panel rows; found {len(panel)}")
    years_by_row, targets_by_row, output_by_row = _lookups(panel)
    panel_rows = panel["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(panel_rows, base.row_indexes):
        raise ValueError("panel and prompt stores do not contain the same sorted row_index set")
    years = sorted(
        int(year) for year in pd.to_datetime(panel["entry_date"], errors="coerce").dt.year.dropna().unique()
    )
    windows = rolling_windows(years)

    prompt_file = base.shards[0].directory / "prompt_tokens.json"
    prompt = json.loads(prompt_file.read_text(encoding="utf-8"))
    stored_ids = np.load(base.shards[0].directory / "prompt_input_ids.npy")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS[args.model], local_files_only=True, use_fast=True,
    )
    mapping_kwargs = {}
    if args.prompt_spec is not None:
        specification = json.loads(args.prompt_spec.read_text(encoding="utf-8"))
        prompt_spec = specification["prompts"][args.prompt_length]
        if str(prompt_spec["text"]) != str(prompt["text"]):
            raise ValueError(f"prompt spec text mismatch for {args.prompt_length}")
        mapping_kwargs = {
            "semantic_phrases": [
                tuple(value) for value in prompt_spec["semantic_phrases"]
            ],
            "require_generic": False,
        }
    semantic_map = validate_tokenizer_mapping(
        tokenizer, str(prompt["text"]), stored_ids, prompt["tokens"], **mapping_kwargs,
    )
    groups = list(semantic_map.group_positions)
    group_positions = [
        np.asarray(semantic_map.group_positions[group], dtype=np.int64) for group in groups
    ]

    final_dir = args.output_root / args.model / args.prompt_length
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite mechanism cache: {final_dir}")
    stage = final_dir.with_name(f".{final_dir.name}.partial.{os.getpid()}")
    stage.mkdir(parents=True)
    variants = (variant, masked_variant)
    group_files = {
        name: np.lib.format.open_memmap(
            stage / f"{name}_group_embeddings.npy", mode="w+", dtype=np.float32,
            shape=(len(panel), len(groups), base.hidden_size),
        ) for name in variants
    }
    prompt_files = {
        name: np.lib.format.open_memmap(
            stage / f"{name}_prompt_mean.npy", mode="w+", dtype=np.float32,
            shape=(len(panel), base.hidden_size),
        ) for name in variants
    }
    body_files = None if args.without_body else {
        name: np.lib.format.open_memmap(
            stage / f"{name}_body_mean.npy", mode="w+", dtype=np.float32,
            shape=(len(panel), base.hidden_size),
        ) for name in variants
    }
    accumulators = {
        name: YearlyTokenMoments(years, base.token_count, base.hidden_size)
        for name in variants
    }
    delta_sum = np.zeros((len(years), base.token_count, base.hidden_size), dtype=np.float64)
    delta_sq = np.zeros_like(delta_sum)
    delta_count = np.zeros(len(years), dtype=np.int64)
    year_position = {year: index for index, year in enumerate(years)}

    try:
        if len(base.shards) != len(masked.shards):
            raise ValueError("masked and unmasked stores have different shard counts")
        for base_shard, masked_shard in zip(base.shards, masked.shards):
            if not np.array_equal(base_shard.row_indexes, masked_shard.row_indexes):
                raise ValueError(f"paired shard row mismatch: {base_shard.directory}")
            row_indexes = base_shard.row_indexes
            positions = output_by_row[row_indexes]
            if (positions < 0).any():
                raise ValueError("embedding shard contains row_index absent from panel")
            base_matrix = np.load(
                base_shard.directory / "prompt_token_embeddings.npy", mmap_mode="r",
            )
            masked_matrix = np.load(
                masked_shard.directory / "prompt_token_embeddings.npy", mmap_mode="r",
            )
            if body_files is not None:
                with np.load(base_shard.directory / "short_pooling.npz") as archive:
                    base_body = np.asarray(archive["body_mean"], dtype=np.float32)
                with np.load(masked_shard.directory / "short_pooling.npz") as archive:
                    masked_body = np.asarray(archive["body_mean"], dtype=np.float32)
                if len(base_body) != len(row_indexes) or len(masked_body) != len(row_indexes):
                    raise ValueError("body pooling rows do not align with prompt-token shard")
                body_files[variant][positions] = base_body
                body_files[masked_variant][positions] = masked_body
            for start in range(0, len(row_indexes), args.batch_size):
                stop = min(start + args.batch_size, len(row_indexes))
                rows = row_indexes[start:stop]
                output_positions = positions[start:stop]
                values = {
                    variant: np.asarray(base_matrix[start:stop], dtype=np.float32),
                    masked_variant: np.asarray(masked_matrix[start:stop], dtype=np.float32),
                }
                target_values = {name: lookup[rows] for name, lookup in targets_by_row.items()}
                row_years = years_by_row[rows]
                dated = row_years > 0
                for name in variants:
                    if dated.any():
                        accumulators[name].update(
                            values[name][dated], row_years[dated],
                            {target: numeric[dated] for target, numeric in target_values.items()},
                        )
                    group_files[name][output_positions] = _group_mean(
                        values[name], group_positions,
                    )
                    prompt_files[name][output_positions] = values[name].mean(
                        axis=1, dtype=np.float32,
                    )
                delta = values[masked_variant].astype(np.float64) - values[variant]
                for year in np.unique(row_years[dated]):
                    selected = delta[row_years == year]
                    index = year_position[int(year)]
                    delta_sum[index] += selected.sum(axis=0, dtype=np.float64)
                    delta_sq[index] += np.square(selected).sum(axis=0, dtype=np.float64)
                    delta_count[index] += len(selected)
            del base_matrix, masked_matrix

        for name in variants:
            group_files[name].flush(); prompt_files[name].flush()
            if body_files is not None:
                body_files[name].flush()
            accumulators[name].save(stage / f"{name}_token_moments.npz")
        np.savez_compressed(
            stage / "mask_delta_moments.npz", years=np.asarray(years, dtype=np.int32),
            delta_sum=delta_sum, delta_sq=delta_sq, delta_count=delta_count,
        )

        topk_reports = {}
        if not args.without_topk:
            topk_positions = {}
            for name in variants:
                selected, report = _fisher_topk(
                    stage / f"{name}_token_moments.npz", windows, args.topk,
                )
                topk_positions[name] = selected
                topk_reports[name] = report
            for name in variants:
                topk_shape = (
                    len(TARGET_COLUMNS), len(windows), len(panel), base.hidden_size,
                )
                topk_files = np.lib.format.open_memmap(
                    stage / f"{name}_top{args.topk}_by_window.npy", mode="w+",
                    dtype=np.float32, shape=topk_shape,
                )
                store = base if name == variant else masked
                for batch in store.iter_value_batches(batch_size=args.batch_size):
                    output_positions = output_by_row[batch.row_indexes]
                    for target_index in range(len(TARGET_COLUMNS)):
                        for window_index in range(len(windows)):
                            selected = topk_positions[name][target_index, window_index]
                            topk_files[target_index, window_index, output_positions] = (
                                batch.values[:, selected, :].mean(axis=1, dtype=np.float32)
                            )
                topk_files.flush()
                del topk_files

        metadata = panel[["row_index", "entry_date", "stock_id", *TARGET_COLUMNS]].copy()
        metadata["stock_id"] = metadata["stock_id"].astype(str).str.zfill(6)
        metadata["industry"] = "unknown"
        if args.industry_map is not None and args.industry_map.is_file():
            industry = pd.read_csv(args.industry_map, dtype={"stock_id": str})
            required = {"stock_id", "industry"}
            if not required.issubset(industry.columns):
                raise ValueError("industry map must contain stock_id and industry")
            industry = industry[["stock_id", "industry"]].drop_duplicates("stock_id")
            lookup = industry.set_index(industry["stock_id"].str.zfill(6))["industry"]
            mapped = metadata["stock_id"].map(lookup)
            metadata["industry"] = mapped.fillna("unknown").astype(str)
        metadata.to_parquet(stage / "rows.parquet", index=False)
        report = {
            "format_version": "prompt_mechanism_cache_v1",
            "model": args.model, "prompt_length": args.prompt_length,
            "variants": list(variants), "rows": len(panel),
            "token_count": base.token_count, "hidden_size": base.hidden_size,
            "years": years, "windows": windows,
            "topk": None if args.without_topk else args.topk,
            "has_body": not args.without_body,
            "groups": groups,
            "group_positions": {
                group: list(semantic_map.group_positions[group]) for group in groups
            },
            "position_groups": list(semantic_map.position_groups),
            "prompt_text": semantic_map.prompt_text,
            "prompt_tokens": list(semantic_map.tokens),
            "prompt_offsets": [list(value) for value in semantic_map.offsets],
            "topk_positions": topk_reports,
            "inputs": {
                "embedding_root": str(args.embedding_root.resolve()),
                "panel": str(args.panel.resolve()),
                "industry_map": str(args.industry_map.resolve()) if args.industry_map else None,
                "expected_shards": args.expected_shards,
                "prompt_spec": str(args.prompt_spec.resolve()) if args.prompt_spec else None,
            },
        }
        atomic_json(stage / "manifest.json", report)
        (stage / "COMPLETED").write_text("prompt_mechanism_cache_v1\n", encoding="utf-8")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(final_dir)
        return {**report, "output": str(final_dir)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt-spec", type=Path)
    parser.add_argument(
        "--industry-map", type=Path,
        default=Path("data/stock_universe_paper_1000.csv"),
    )
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--prompt-length", choices=PROMPT_LENGTHS, required=True)
    parser.add_argument("--expected-shards", type=int, default=64)
    parser.add_argument("--expected-rows", type=int, default=75894)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--without-body", action="store_true")
    parser.add_argument("--without-topk", action="store_true")
    args = parser.parse_args()
    if min(args.expected_shards, args.expected_rows, args.batch_size, args.topk) < 1:
        raise ValueError("counts, batch size, and topk must be positive")
    report = build_cache(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
