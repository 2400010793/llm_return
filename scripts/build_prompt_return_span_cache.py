"""Extract one semantic prompt-span embedding into an aligned mmap cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import MODEL_PATHS
from src.analysis.prompt_token_mechanisms import validate_tokenizer_mapping
from src.data.prompt_token_embeddings import PromptTokenEmbeddingStore
from src.evaluation.artifacts import atomic_json


def build(args: argparse.Namespace) -> dict[str, object]:
    base_variant = args.prompt_length
    masked_variant = f"masked_{args.prompt_length}"
    stores = {
        base_variant: PromptTokenEmbeddingStore(
            args.embedding_root, model=args.model, variant=base_variant,
            expected_shards=args.expected_shards, expected_rows=args.expected_rows,
        ),
        masked_variant: PromptTokenEmbeddingStore(
            args.embedding_root, model=args.model, variant=masked_variant,
            expected_shards=args.expected_shards, expected_rows=args.expected_rows,
        ),
    }
    base = stores[base_variant]
    masked = stores[masked_variant]
    if not np.array_equal(base.row_indexes, masked.row_indexes):
        raise ValueError("masked and unmasked stores do not align")
    source = base.shards[0].directory
    prompt = json.loads((source / "prompt_tokens.json").read_text(encoding="utf-8"))
    stored_ids = np.load(source / "prompt_input_ids.npy")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS[args.model], local_files_only=True, use_fast=True,
    )
    mapping = validate_tokenizer_mapping(
        tokenizer, str(prompt["text"]), stored_ids, prompt["tokens"],
        semantic_phrases=((args.semantic_name, args.phrase),), require_generic=False,
    )
    positions = np.asarray(mapping.group_positions[args.semantic_name], dtype=np.int64)
    destination = args.output_root / args.model / args.prompt_length / args.semantic_name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite span cache: {destination}")
    stage = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    stage.mkdir(parents=True)
    files = {
        variant: np.lib.format.open_memmap(
            stage / f"{variant}_{args.semantic_name}.npy", mode="w+", dtype=np.float32,
            shape=(base.rows, base.hidden_size),
        ) for variant in stores
    }
    output_by_row = np.full(base.max_row_index + 1, -1, dtype=np.int64)
    output_by_row[base.row_indexes] = np.arange(base.rows, dtype=np.int64)
    try:
        for variant, store in stores.items():
            for batch in store.iter_value_batches(batch_size=args.batch_size):
                output_positions = output_by_row[batch.row_indexes]
                files[variant][output_positions] = batch.values[:, positions, :].mean(
                    axis=1, dtype=np.float32,
                )
            files[variant].flush()
        report = {
            "format_version": "prompt_semantic_span_cache_v2",
            "model": args.model, "prompt_length": args.prompt_length,
            "variants": list(stores), "rows": base.rows,
            "hidden_size": base.hidden_size, "semantic_name": args.semantic_name,
            "phrase": args.phrase,
            "positions_zero_based": positions.tolist(),
            "tokens": [mapping.tokens[index] for index in positions],
            "offsets": [list(mapping.offsets[index]) for index in positions],
            "prompt_text": mapping.prompt_text,
        }
        atomic_json(stage / "manifest.json", report)
        (stage / "COMPLETED").write_text("prompt_return_span_cache_v1\n", encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(destination)
        return {**report, "output": str(destination)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--prompt-length", choices=("short", "long"), required=True)
    parser.add_argument("--expected-shards", type=int, default=64)
    parser.add_argument("--expected-rows", type=int, default=75894)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--semantic-name", default="return_span")
    parser.add_argument("--phrase", default="收益")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
