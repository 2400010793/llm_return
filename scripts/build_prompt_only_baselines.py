"""Encode fixed Sina prompts without an article for contextual-shift baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import MODEL_PATHS
from src.analysis.prompt_token_mechanisms import validate_tokenizer_mapping
from src.evaluation.artifacts import atomic_json


MODELS = ("roberta", "bge_m3", "ckip_bert", "xlm_roberta_large")
PROMPT_LENGTHS = ("short", "long")


def encode_one(model_name: str, prompt_length: str, args: argparse.Namespace) -> dict[str, object]:
    source = args.embedding_root / "shard-0" / model_name / prompt_length
    prompt = json.loads((source / "prompt_tokens.json").read_text(encoding="utf-8"))
    stored_ids = np.load(source / "prompt_input_ids.npy")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(MODEL_PATHS[model_name])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True,
    )
    semantic_phrases = None
    if args.prompt_spec is not None:
        specification = json.loads(args.prompt_spec.read_text(encoding="utf-8"))
        prompt_spec = specification["prompts"][prompt_length]
        if str(prompt_spec["text"]) != str(prompt["text"]):
            raise ValueError(f"prompt spec text mismatch for {prompt_length}")
        semantic_phrases = [tuple(value) for value in prompt_spec["semantic_phrases"]]
    mapping_kwargs = {}
    if semantic_phrases is not None:
        mapping_kwargs = {
            "semantic_phrases": semantic_phrases,
            "require_generic": False,
        }
    mapping = validate_tokenizer_mapping(
        tokenizer, str(prompt["text"]), stored_ids, prompt["tokens"], **mapping_kwargs,
    )
    empty = tokenizer("", add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if len(empty) < 2:
        raise ValueError(f"{model_name} tokenizer lacks boundary special tokens")
    ids = np.asarray([[int(empty[0]), *stored_ids.astype(int).tolist(), int(empty[-1])]])
    attention = np.ones_like(ids)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(args.device).eval()
    with torch.inference_mode():
        hidden = model(
            input_ids=torch.from_numpy(ids).to(args.device),
            attention_mask=torch.from_numpy(attention).to(args.device),
        ).last_hidden_state[:, 1:1 + len(stored_ids), :]
    values = hidden[0].cpu().numpy().astype(np.float32)
    destination = args.output_root / model_name / f"{prompt_length}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, prompt_only=values, input_ids=stored_ids.astype(np.int32),
            offsets=np.asarray(mapping.offsets, dtype=np.int32),
        )
    temporary.replace(destination)
    del model
    return {
        "model": model_name, "prompt_length": prompt_length,
        "model_path": str(model_path), "source": str(source),
        "output": str(destination), "shape": list(values.shape),
        "semantic_groups": {
            group: list(positions) for group, positions in mapping.group_positions.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt-spec", type=Path)
    parser.add_argument("--model", choices=MODELS, action="append")
    parser.add_argument("--prompt-length", choices=PROMPT_LENGTHS, action="append")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows = []
    for model in args.model or MODELS:
        for prompt_length in args.prompt_length or PROMPT_LENGTHS:
            rows.append(encode_one(model, prompt_length, args))
    report = {
        "format_version": "prompt_only_baseline_v1", "device": args.device,
        "embedding_root": str(args.embedding_root.resolve()), "baselines": rows,
    }
    atomic_json(args.output_root / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
