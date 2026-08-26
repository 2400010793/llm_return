"""Build tokenizer-specific semantic maps for the minimal v2 prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import MODEL_PATHS
from src.analysis.prompt_token_mechanisms import build_semantic_token_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["roberta", "bge_m3"])
    args = parser.parse_args()

    from transformers import AutoTokenizer

    specification = json.loads(args.spec.read_text(encoding="utf-8"))
    results = []
    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATHS[model_name], local_files_only=True, use_fast=True,
        )
        for prompt_key, prompt_spec in specification["prompts"].items():
            text = str(prompt_spec["text"])
            encoded = tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True,
            )
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
            phrases = [tuple(item) for item in prompt_spec["semantic_phrases"]]
            mapping = build_semantic_token_map(
                text, tokens, encoded["offset_mapping"],
                semantic_phrases=phrases, require_generic=False,
            )
            phrase_audit = []
            for group, phrase in phrases:
                start = text.index(phrase)
                stop = start + len(phrase)
                positions = mapping.group_positions[group]
                covered = set()
                for position in positions:
                    token_start, token_stop = mapping.offsets[position]
                    covered.update(range(max(start, token_start), min(stop, token_stop)))
                expected = {index for index in range(start, stop) if not text[index].isspace()}
                if not expected.issubset(covered):
                    raise ValueError(
                        f"{model_name}/{prompt_key}/{group} does not cover its phrase"
                    )
                phrase_audit.append({
                    "group": group, "phrase": phrase, "char_span": [start, stop],
                    "token_positions": list(positions),
                    "tokens": [tokens[position] for position in positions],
                    "offset_slices": [
                        text[mapping.offsets[position][0]:mapping.offsets[position][1]]
                        for position in positions
                    ],
                })
            results.append({
                "model": model_name, "model_path": MODEL_PATHS[model_name],
                "prompt_key": prompt_key, "text": text,
                "token_count": len(tokens), "input_ids": encoded["input_ids"],
                "tokens": tokens,
                "offsets": [list(value) for value in encoded["offset_mapping"]],
                "position_groups": list(mapping.position_groups),
                "group_positions": {
                    group: list(positions)
                    for group, positions in mapping.group_positions.items()
                },
                "phrase_audit": phrase_audit,
            })
    report = {
        "format_version": "minimal_prompt_tokenizer_audit_v1",
        "prompt_spec": str(args.spec), "all_valid": True, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
