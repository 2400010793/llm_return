"""Encode one position-matched no-prompt or semantic replacement control shard."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import (
    DEFAULT_MAX_LENGTH,
    MODEL_PATHS,
    components,
    iter_records,
    masked_mean,
    sequence_audit,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite control output: {args.output_dir}")
    rows = list(iter_records(args.input, args.limit))
    if not rows:
        raise ValueError("control input shard is empty")
    specs = json.loads(args.control_selection.read_text(encoding="utf-8"))["control_specs"]
    spec = specs[args.spec_index]
    if spec["model"] != args.model or spec["source_variant"] != args.source_variant:
        raise ValueError("control manifest row does not match selected spec")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(MODEL_PATHS[args.model])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(args.device).eval()
    prompt_text = components(rows[0], args.source_variant)[0]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    for row in rows[1:]:
        if tokenizer(components(row, args.source_variant)[0], add_special_tokens=False)["input_ids"] != prompt_ids:
            raise ValueError("source prompt changes within control shard")
    empty = tokenizer("", add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if len(empty) < 2 or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must provide boundary and padding tokens")
    bos_id, eos_id, pad_id = int(empty[0]), int(empty[-1]), int(tokenizer.pad_token_id)
    replacement_id = tokenizer.mask_token_id
    if replacement_id is None:
        replacement_id = tokenizer.unk_token_id
    if replacement_id is None:
        raise ValueError("tokenizer has neither mask nor unknown replacement token")
    controlled_prompt = list(map(int, prompt_ids))
    prompt_attention = [1] * len(controlled_prompt)
    if spec["mode"] == "no_prompt_position_matched":
        controlled_prompt = [pad_id] * len(controlled_prompt)
        prompt_attention = [0] * len(controlled_prompt)
    elif spec["mode"] == "replace_positions":
        for position in spec["positions"]:
            if position < 0 or position >= len(controlled_prompt):
                raise ValueError(f"control position outside prompt: {position}")
            controlled_prompt[position] = int(replacement_id)
    else:
        raise ValueError(f"unknown control mode: {spec['mode']}")

    max_length = args.max_length or DEFAULT_MAX_LENGTH[args.model]
    sequences = []
    metadata = []
    for row in rows:
        _, title, body = components(row, args.source_variant)
        ids = [bos_id, *controlled_prompt]
        segments = [-1, *([0] * len(controlled_prompt))]
        valid = [1, *prompt_attention]
        for segment_id, text in ((1, title), (2, body)):
            part = tokenizer(text, add_special_tokens=False)["input_ids"]
            ids.extend(part); segments.extend([segment_id] * len(part)); valid.extend([1] * len(part))
        ids.append(eos_id); segments.append(-1); valid.append(1)
        original_count = len(ids)
        ids, segments, valid = ids[:max_length], segments[:max_length], valid[:max_length]
        sequences.append((ids, segments, valid))
        metadata.append({
            "row_index": int(row["row_index"]), "document_id": row.get("document_id"),
            "article_id": row.get("article_id"), "variant": args.output_variant,
            "source_variant": args.source_variant, "control_mode": spec["mode"],
            "semantic_group": spec["semantic_group"], "is_placebo": spec["is_placebo"],
            "original_token_count": original_count, "saved_token_count": len(ids),
            "truncated": original_count > max_length,
            **sequence_audit(ids, segments, valid),
        })

    output = args.output_dir
    output.mkdir(parents=True)
    names = ("prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean", "cls")
    pooled = {name: [] for name in names}
    try:
        with torch.inference_mode():
            for start in range(0, len(sequences), args.batch_size):
                batch = sequences[start:start + args.batch_size]
                width = max(len(item[0]) for item in batch)
                input_ids = np.full((len(batch), width), pad_id, dtype=np.int64)
                attention = np.zeros((len(batch), width), dtype=np.int64)
                segment_ids = np.full((len(batch), width), -2, dtype=np.int8)
                for index, (ids, segments, valid) in enumerate(batch):
                    input_ids[index, :len(ids)] = ids
                    attention[index, :len(ids)] = valid
                    segment_ids[index, :len(ids)] = segments
                attention_t = torch.from_numpy(attention).to(args.device)
                hidden = model(
                    input_ids=torch.from_numpy(input_ids).to(args.device),
                    attention_mask=attention_t,
                ).last_hidden_state
                segments_t = torch.from_numpy(segment_ids).to(args.device)
                visible = attention_t.bool()
                masks = {
                    "prompt": (segments_t == 0) & visible,
                    "title": (segments_t == 1) & visible,
                    "body": (segments_t == 2) & visible,
                }
                masks["title_body"] = masks["title"] | masks["body"]
                masks["full"] = masks["prompt"] | masks["title_body"]
                values = {
                    "prompt_mean": masked_mean(hidden, masks["prompt"]),
                    "title_mean": masked_mean(hidden, masks["title"]),
                    "body_mean": masked_mean(hidden, masks["body"]),
                    "title_body_mean": masked_mean(hidden, masks["title_body"]),
                    "full_mean": masked_mean(hidden, masks["full"]),
                    "cls": hidden[:, 0, :],
                }
                for name, value in values.items():
                    pooled[name].append(value.cpu().numpy().astype(np.float32))
        arrays = {name: np.concatenate(parts) for name, parts in pooled.items()}
        np.savez_compressed(output / "short_pooling.npz", **arrays)
        pd.DataFrame(metadata).to_json(
            output / "metadata.jsonl", orient="records", lines=True, force_ascii=False,
        )
        summary = {
            "format_version": "prompt_position_control_v1", "model": args.model,
            "variant": args.output_variant, "control_id": spec["control_id"],
            "source_variant": args.source_variant,
            "prompt_condition": spec["mode"], "control_spec": spec,
            "rows": len(rows), "max_length": max_length,
            "hidden_size": int(model.config.hidden_size),
            "outputs": {name: list(value.shape) for name, value in arrays.items()},
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return summary
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-selection", type=Path, required=True)
    parser.add_argument("--spec-index", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-variant", choices=("long", "masked_long"), required=True)
    parser.add_argument("--output-variant", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
