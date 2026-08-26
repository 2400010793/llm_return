"""Preflight the clean masked-short prompt axes with a local tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import MODEL_PATHS, components, iter_records
from src.data.aligned_prompts import audit_aligned_prompts, load_aligned_prompt_config


def _probe_masked_sequences(rows, audit, tokenizer, max_length: int) -> dict[str, object]:
    if not rows:
        raise ValueError("at least one input row is required for the synthetic probe")
    row = rows[0]
    prompt_metrics = []
    special = tokenizer("", add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if len(special) < 2 or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must provide BOS/EOS and PAD")
    bos_id, eos_id = int(special[0]), int(special[-1])
    prompt_token_count = None
    for prompt_id, prompt_audit in audit["prompts"].items():
        prompt = prompt_audit["text"]
        _, title, body = components(row, "masked_short")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        title_ids = tokenizer(title, add_special_tokens=False)["input_ids"]
        body_ids = tokenizer(body, add_special_tokens=False)["input_ids"]
        sequence = [bos_id] + prompt_ids + title_ids + body_ids + [eos_id]
        visible = sequence[:max_length]
        body_start = 1 + len(prompt_ids) + len(title_ids)
        visible_body = max(0, len(visible) - body_start - (1 if len(visible) == max_length and visible[-1] == eos_id else 0))
        current = {
            "prompt_id": prompt_id,
            "prompt_token_count": len(prompt_ids),
            "title_token_count": len(title_ids),
            "body_start_zero_based": body_start,
            "visible_body_tokens": visible_body,
            "original_token_count": len(sequence),
            "saved_token_count": len(visible),
            "sequence_truncated": len(sequence) > max_length,
            "masked_stock_id_present": str(row.get("stock_id_alignment", "")) in (title + body),
        }
        prompt_metrics.append(current)
        prompt_token_count = len(prompt_ids) if prompt_token_count is None else prompt_token_count
    comparable = ("prompt_token_count", "title_token_count", "body_start_zero_based", "visible_body_tokens", "saved_token_count", "sequence_truncated")
    by_axis = {}
    for metric in prompt_metrics:
        axis = audit["prompts"][metric["prompt_id"]]["axis"]
        by_axis.setdefault(axis, []).append(metric)
    axis_mismatches = {}
    for axis, metrics in by_axis.items():
        mismatches = {
            field: sorted({metric[field] for metric in metrics})
            for field in comparable
            if len({metric[field] for metric in metrics}) != 1
        }
        if mismatches:
            axis_mismatches[axis] = mismatches
    if axis_mismatches:
        raise ValueError(f"masked-short synthetic probe is not aligned within axis: {axis_mismatches}")
    if any(metric["masked_stock_id_present"] for metric in prompt_metrics):
        raise ValueError("stock_id_alignment remains in masked title/body")
    return {
        "row_index": row.get("row_index"),
        "max_length": max_length,
        "prompt_metrics": prompt_metrics,
        "aligned": True,
        "alignment_scope": "within_axis",
        "axis_token_budgets": {
            axis: sorted({metric["prompt_token_count"] for metric in metrics})
            for axis, metrics in by_axis.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model", choices=tuple(MODEL_PATHS), default="roberta")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = load_aligned_prompt_config(args.config)
    model_path = Path(MODEL_PATHS[args.model])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    audit = audit_aligned_prompts(config, tokenizer)
    if args.input:
        rows = list(iter_records(args.input, 1))
        max_length = args.max_length or 512
        audit["synthetic_probe"] = _probe_masked_sequences(rows, audit, tokenizer, max_length)
    audit["config_path"] = str(args.config)
    audit["model_path"] = str(model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "axes": list(audit["axes"]), "prompts": len(audit["prompts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
