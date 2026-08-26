"""Replicate short-prompt token mechanism metrics on the 903,665-row CNInfo panel."""

from __future__ import annotations

import argparse
import json
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
    build_semantic_token_map,
    rolling_windows,
    token_metrics_from_moments,
)
from src.data.prompt_token_embeddings import PromptTokenEmbeddingStore


CNINFO_PHRASES = (
    ("target_stock", "这只股票"),
    ("future_horizon", "未来"),
    ("direction", "涨跌"),
    ("article_instruction", "请基于下面的文章确定"),
)


def _write_report(
    args: argparse.Namespace,
    *,
    rows: int,
    years: list[int],
    direction_group_rows: int,
) -> dict[str, object]:
    report = {
        "format_version": "cninfo_prompt_mechanism_replication_v1",
        "model": args.model,
        "rows": int(rows),
        "years": [int(year) for year in years],
        "variants": ["short", "masked_short"],
        "targets": list(TARGET_COLUMNS),
        "direction_group_rows": int(direction_group_rows),
        "interpretation": "external short-prompt representation replication; not a long-prompt category or causal test",
    }
    (args.output_root / f"{args.model}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (args.output_root / f"{args.model}_COMPLETED").write_text("ok\n", encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    variants = ("short", "masked_short")
    stores = {
        variant: PromptTokenEmbeddingStore(
            args.embedding_root, model=args.model, variant=variant,
            expected_shards=args.expected_shards, expected_rows=args.expected_rows,
        ) for variant in variants
    }
    base, masked = stores["short"], stores["masked_short"]
    if not np.array_equal(base.row_indexes, masked.row_indexes):
        raise ValueError("CNInfo masked/unmasked row sets differ")
    panel = pd.read_parquet(
        args.panel, columns=["row_index", "entry_date", *TARGET_COLUMNS],
    ).sort_values("row_index", kind="stable").reset_index(drop=True)
    if len(panel) != args.expected_rows or not np.array_equal(
        panel["row_index"].to_numpy(dtype=np.int64), base.row_indexes,
    ):
        raise ValueError("CNInfo panel does not align with prompt-token rows")
    dates = pd.to_datetime(panel["entry_date"], errors="coerce")
    years = sorted(dates.dt.year.dropna().astype(int).unique())
    windows = rolling_windows(years)
    args.output_root.mkdir(parents=True, exist_ok=True)
    token_output = args.output_root / f"{args.model}_token_metrics.parquet"
    group_output = args.output_root / f"{args.model}_group_metrics.parquet"
    if token_output.stat().st_size > 0 if token_output.exists() else False:
        if group_output.stat().st_size > 0 if group_output.exists() else False:
            existing_groups = pd.read_parquet(group_output, columns=["semantic_group"])
            return _write_report(
                args,
                rows=len(panel),
                years=years,
                direction_group_rows=existing_groups["semantic_group"].eq("direction").sum(),
            )
    size = int(panel["row_index"].max()) + 1
    year_lookup = np.full(size, -1, dtype=np.int32)
    valid = dates.notna().to_numpy()
    row_indexes = panel["row_index"].to_numpy(dtype=np.int64)
    year_lookup[row_indexes[valid]] = dates.loc[valid].dt.year.to_numpy(dtype=np.int32)
    target_lookup = {}
    for target in TARGET_COLUMNS:
        values = np.full(size, np.nan, dtype=np.float64)
        values[row_indexes] = pd.to_numeric(panel[target], errors="coerce").to_numpy(dtype=float)
        target_lookup[target] = values
    accumulators = {
        variant: YearlyTokenMoments(years, base.token_count, base.hidden_size)
        for variant in variants
    }
    delta_sum = np.zeros((len(years), base.token_count, base.hidden_size), dtype=np.float64)
    delta_count = np.zeros(len(years), dtype=np.int64)
    year_position = {year: index for index, year in enumerate(years)}
    base_batches = base.iter_value_batches(batch_size=args.batch_size)
    masked_batches = masked.iter_value_batches(batch_size=args.batch_size)
    for base_batch, masked_batch in zip(base_batches, masked_batches):
        if not np.array_equal(base_batch.row_indexes, masked_batch.row_indexes):
            raise ValueError("CNInfo paired prompt batches are misaligned")
        rows = base_batch.row_indexes
        row_years = year_lookup[rows]
        dated = row_years > 0
        targets = {target: values[rows] for target, values in target_lookup.items()}
        if dated.any():
            for variant, batch in (("short", base_batch), ("masked_short", masked_batch)):
                accumulators[variant].update(
                    batch.values[dated], row_years[dated],
                    {target: values[dated] for target, values in targets.items()},
                )
            delta = (
                masked_batch.values[dated].astype(np.float64)
                - base_batch.values[dated].astype(np.float64)
            )
            for year in np.unique(row_years[dated]):
                selected = delta[row_years[dated] == year]
                position = year_position[int(year)]
                delta_sum[position] += selected.sum(axis=0, dtype=np.float64)
                delta_count[position] += len(selected)

    prompt = json.loads(
        (base.shards[0].directory / "prompt_tokens.json").read_text(encoding="utf-8")
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS[args.model], local_files_only=True, use_fast=True,
    )
    encoded = tokenizer(
        prompt["text"], add_special_tokens=False, return_offsets_mapping=True,
    )
    mapping = build_semantic_token_map(
        prompt["text"], prompt["tokens"], encoded["offset_mapping"],
        semantic_phrases=CNINFO_PHRASES,
    )
    with np.load(args.baseline_root / args.model / "short.npz") as archive:
        baseline = np.asarray(archive["prompt_only"], dtype=np.float64)
    output_rows = []
    group_rows = []
    for variant in variants:
        moment_path = args.output_root / f".{args.model}_{variant}_moments.npz"
        accumulators[variant].save(moment_path)
        with np.load(moment_path) as archive:
            for target in TARGET_COLUMNS:
                for window in windows:
                    selected_years = [year_position[int(year)] for year in window["fit_years"]]
                    mask_delta = delta_sum[selected_years].sum(axis=0) / delta_count[selected_years].sum()
                    metrics = token_metrics_from_moments(
                        aggregate_moments(archive, window["fit_years"], target=target),
                        prompt_only=baseline, mask_delta_mean=mask_delta,
                    )
                    order = np.lexsort((np.arange(base.token_count), -metrics["fisher"]))
                    rank = np.empty(base.token_count, dtype=np.int32)
                    rank[order] = np.arange(1, base.token_count + 1)
                    fold = pd.DataFrame({
                        "model": args.model, "variant": variant, "target": target,
                        "test_year": int(window["test_year"]),
                        "position_zero_based": np.arange(base.token_count),
                        "token": prompt["tokens"], "semantic_group": mapping.position_groups,
                        "fisher": metrics["fisher"], "fisher_rank": rank,
                        "context_cosine_distance": metrics["context_cosine_distance"],
                        "context_standardized_l2": metrics["context_standardized_l2"],
                        "mask_delta_l2": metrics["mask_delta_l2"],
                    })
                    output_rows.append(fold)
                    groups = fold.groupby("semantic_group", sort=False).agg(
                        token_count=("token", "size"), fisher_mean=("fisher", "mean"),
                        fisher_rank_best=("fisher_rank", "min"),
                        context_standardized_l2_mean=("context_standardized_l2", "mean"),
                        mask_delta_l2_mean=("mask_delta_l2", "mean"),
                    ).reset_index()
                    group_rows.append(groups.assign(
                        model=args.model, variant=variant, target=target,
                        test_year=int(window["test_year"]),
                    ))
        moment_path.unlink()
    tokens = pd.concat(output_rows, ignore_index=True)
    groups = pd.concat(group_rows, ignore_index=True)
    tokens.to_parquet(args.output_root / f"{args.model}_token_metrics.parquet", index=False)
    groups.to_parquet(args.output_root / f"{args.model}_group_metrics.parquet", index=False)
    direction = groups[groups["semantic_group"].eq("direction")]
    return _write_report(
        args, rows=len(panel), years=years, direction_group_rows=len(direction),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--expected-shards", type=int, default=64)
    parser.add_argument("--expected-rows", type=int, default=903665)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
