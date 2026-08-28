"""Save unpooled token hidden states for short CNInfo prompts.

No chunking is used. Prompt, title, and body are tokenized separately and
then concatenated. The saved arrays retain token order, offsets, and segment
labels. Defaults are 512 tokens for RoBERTa and 1000 for BGE-M3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_PATHS = {
    "roberta": "/home/team/llm_return/models/chinese-roberta-wwm-ext",
    "bge_m3": "/home/team/llm_return/models/bge-m3",
}
DEFAULT_MAX_LENGTH = {"roberta": 512, "bge_m3": 1000}


def iter_records(source: Path, limit: int | None):
    files = sorted(source.rglob("part-*.jsonl")) if source.is_dir() else [source]
    count = 0
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                count += 1
                if limit is not None and count >= limit:
                    return


def components(row: dict, variant: str) -> tuple[str, str, str]:
    masked = variant == "masked_short"
    prefix = "masked_short" if masked else "short"
    full_key = "text_input_5_masked_short" if masked else "text_input_2_short"
    full = str(row.get(full_key, "") or "")
    prompt = str(row.get(f"{prefix}_prompt", "") or "")
    title = str(row.get(f"{prefix}_title", "") or "")
    body = str(row.get(f"{prefix}_body", "") or "")
    # Backward-compatible fallback for parts generated before component fields.
    if not prompt:
        marker = "文章："
        pos = full.find(marker)
        prompt = full[:pos + len(marker)] if pos >= 0 else full
    if not body:
        body = full[len(prompt):] if full.startswith(prompt) else full
    return prompt, title, body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_PATHS), required=True)
    parser.add_argument("--variant", choices=("short", "masked_short"), required=True)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--prompt-length", type=int, default=None,
                        help="Fixed prompt token slots, excluding model special tokens")
    args = parser.parse_args()
    max_length = args.max_length or DEFAULT_MAX_LENGTH[args.model]
    if args.model == "roberta" and max_length > 512:
        raise ValueError("Chinese RoBERTa model limit is 512 tokens")
    if args.model == "bge_m3" and max_length > 1000:
        raise ValueError("This experiment caps BGE-M3 at 1000 tokens")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = MODEL_PATHS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    special_ids = tokenizer("", add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if len(special_ids) < 2:
        raise ValueError("tokenizer must provide start and end special tokens")
    bos_id, eos_id = special_ids[0], special_ids[-1]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad token")

    rows = list(iter_records(args.input, args.rows))
    if not rows:
        raise ValueError("input contains no records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_slots = args.prompt_length
    sequences = []
    metadata = []

    for row in rows:
        texts = components(row, args.variant)
        encoded_parts = []
        for segment_id, text in enumerate(texts):
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = list(encoded["input_ids"])
            offsets = list(encoded["offset_mapping"])
            if segment_id == 0:
                if prompt_slots is None:
                    prompt_slots = len(ids)
                if len(ids) > prompt_slots:
                    ids, offsets = ids[:prompt_slots], offsets[:prompt_slots]
                elif len(ids) < prompt_slots:
                    missing = prompt_slots - len(ids)
                    ids += [pad_id] * missing
                    offsets += [(0, 0)] * missing
            encoded_parts.append((ids, offsets, segment_id))

        ids = [bos_id]
        offsets = [(0, 0)]
        segments = [-1]
        valid = [1]
        for part_ids, part_offsets, segment_id in encoded_parts:
            ids.extend(part_ids)
            offsets.extend(part_offsets)
            segments.extend([segment_id] * len(part_ids))
            # Prompt padding occupies fixed slots but is not a model-visible token.
            valid.extend([0 if segment_id == 0 and token_id == pad_id else 1 for token_id in part_ids])
        ids.append(eos_id)
        offsets.append((0, 0))
        segments.append(-1)
        valid.append(1)

        original_count = len(ids)
        ids, offsets, segments, valid = (
            ids[:max_length], offsets[:max_length], segments[:max_length], valid[:max_length]
        )
        sequences.append((ids, offsets, segments, valid))
        metadata.append({
            "row_index": row.get("row_index"),
            "document_id": row.get("document_id"),
            "variant": args.variant,
            "original_token_count": original_count,
            "saved_token_count": len(ids),
            "truncated": original_count > max_length,
            "prompt_token_slots": prompt_slots,
        })

    hidden_batches = []
    token_batches = []
    with torch.inference_mode():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start:start + args.batch_size]
            width = max(len(x[0]) for x in batch)
            ids = np.full((len(batch), width), pad_id, dtype=np.int64)
            attention = np.zeros((len(batch), width), dtype=np.int64)
            segments = np.full((len(batch), width), -2, dtype=np.int8)
            valid = np.zeros((len(batch), width), dtype=np.int8)
            offsets = np.zeros((len(batch), width, 2), dtype=np.int32)
            for i, (part_ids, part_offsets, part_segments, part_valid) in enumerate(batch):
                n = len(part_ids)
                ids[i, :n] = part_ids
                attention[i, :n] = part_valid
                segments[i, :n] = part_segments
                valid[i, :n] = part_valid
                offsets[i, :n] = part_offsets
            output = model(
                input_ids=torch.from_numpy(ids).to(device),
                attention_mask=torch.from_numpy(attention).to(device),
            ).last_hidden_state
            hidden_batch = output.cpu().numpy().astype(np.float32)
            if hidden_batch.shape[1] < max_length:
                padded_hidden = np.zeros(
                    (hidden_batch.shape[0], max_length, hidden_batch.shape[2]),
                    dtype=np.float32,
                )
                padded_hidden[:, :hidden_batch.shape[1]] = hidden_batch
                hidden_batch = padded_hidden
            hidden_batches.append(hidden_batch)
            token_batches.append((ids, attention, segments, valid, offsets))

    hidden = np.concatenate(hidden_batches, axis=0)
    arrays = []
    for i in range(5):
        padded = []
        for batch in token_batches:
            value = batch[i]
            if value.shape[1] < max_length:
                shape = list(value.shape)
                shape[1] = max_length
                fill = np.zeros(shape, dtype=value.dtype)
                if i == 0:
                    fill[...] = pad_id
                elif i == 2:
                    fill[...] = -2
                fill[:, :value.shape[1]] = value
                value = fill
            padded.append(value)
        arrays.append(np.concatenate(padded, axis=0))
    np.savez_compressed(
        args.output_dir / "token_sequences.npz",
        hidden_states=hidden,
        input_ids=arrays[0],
        attention_mask=arrays[1],
        segment_ids=arrays[2],
        valid_token_mask=arrays[3],
        offsets=arrays[4],
    )
    (args.output_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metadata),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(json.dumps({
        "model": args.model,
        "model_path": model_path,
        "variant": args.variant,
        "rows": len(rows),
        "max_length": max_length,
        "prompt_token_slots": prompt_slots,
        "hidden_shape": list(hidden.shape),
        "segment_ids": {"-1": "special", "0": "prompt", "1": "title", "2": "body"},
        "note": "No chunking. Sequences beyond max_length are truncated at the end.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "hidden_shape": list(hidden.shape),
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
