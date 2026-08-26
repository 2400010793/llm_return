"""Audit all minimal-v2 prompt-token shards and tokenizer alignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import MODEL_PATHS
from src.data.prompt_token_embeddings import PromptTokenEmbeddingStore


HIDDEN_SIZES = {"roberta": 768, "bge_m3": 1024}
VARIANTS = ("short", "masked_short", "long", "masked_long")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=64)
    parser.add_argument("--expected-rows", type=int, default=75894)
    parser.add_argument("--models", nargs="+", default=["roberta", "bge_m3"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    specification = json.loads(args.spec.read_text(encoding="utf-8"))
    results = []
    all_valid = True
    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATHS[model_name], local_files_only=True, use_fast=True,
        )
        for variant in args.variants:
            prompt_key = "long" if "long" in variant else "short"
            text = str(specification["prompts"][prompt_key]["text"])
            ids = np.asarray(
                tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64,
            )
            expected_tokens = tuple(tokenizer.convert_ids_to_tokens(ids.tolist()))
            errors = []
            try:
                store = PromptTokenEmbeddingStore(
                    args.root, model=model_name, variant=variant,
                    expected_shards=args.expected_shards,
                    expected_rows=args.expected_rows,
                    expected_token_count=len(ids),
                    expected_hidden_size=HIDDEN_SIZES[model_name],
                )
                if store.prompt_text != text:
                    errors.append("stored prompt text differs from specification")
                if store.prompt_tokens != expected_tokens:
                    errors.append("stored prompt tokens differ from tokenizer")
                if not np.array_equal(
                    store.row_indexes, np.arange(1, args.expected_rows + 1, dtype=np.int64),
                ):
                    errors.append("row_index coverage is not contiguous 1..expected_rows")
                prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                for shard in store.shards:
                    directory = shard.directory
                    if not (directory / "COMPLETED").is_file():
                        errors.append(f"missing COMPLETED: {directory}")
                    stored_ids = np.load(directory / "prompt_input_ids.npy")
                    if not np.array_equal(stored_ids, ids):
                        errors.append(f"prompt IDs differ: {directory}")
                    summary = json.loads(
                        (directory / "summary.json").read_text(encoding="utf-8")
                    )
                    if summary.get("prompt_sha256") != prompt_hash:
                        errors.append(f"prompt hash differs: {directory}")
                    if summary.get("prompt_key") != prompt_key:
                        errors.append(f"prompt key differs: {directory}")
                rows = store.rows
                shards = len(store.shards)
                shape = [store.token_count, store.hidden_size]
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
                rows, shards, shape = 0, 0, None
            valid = not errors
            all_valid &= valid
            results.append({
                "model": model_name, "variant": variant,
                "prompt_key": prompt_key, "expected_tokens": len(ids),
                "shards": shards, "rows": rows, "token_hidden_shape": shape,
                "valid": valid, "errors": errors,
            })
    report = {
        "format_version": "minimal_prompt_embeddings_audit_v1",
        "root": str(args.root), "prompt_spec": str(args.spec),
        "expected_shards": args.expected_shards,
        "expected_rows": args.expected_rows,
        "variants": args.variants,
        "all_valid": all_valid, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
