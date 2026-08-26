"""Generate CNINFO-style pooled and prompt-token embeddings for one Sina shard."""

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
    masked_max,
    masked_mean,
    sequence_audit,
)


VARIANTS = ("plain", "short", "masked_short", "long", "masked_long")


def encode_variant(rows, *, model, tokenizer, torch, device, variant, output_dir, batch_size, max_length, prompt_text_override=None):
    prompt_text = prompt_text_override or components(rows[0], variant)[0]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if prompt_text_override is None:
        for row in rows[1:]:
            candidate = tokenizer(components(row, variant)[0], add_special_tokens=False)["input_ids"]
            if candidate != prompt_ids:
                raise ValueError(f"prompt token IDs change within variant={variant}")
    prompt_length = len(prompt_ids)
    if variant != "plain" and prompt_length < 1:
        raise ValueError(f"empty prompt for {variant}")

    empty = tokenizer("", add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if len(empty) < 2 or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must provide BOS/EOS and PAD")
    bos_id, eos_id, pad_id = int(empty[0]), int(empty[-1]), int(tokenizer.pad_token_id)
    sequences = []
    metadata = []
    for row in rows:
        ids = [bos_id]
        segments = [-1]
        row_components = components(row, variant)
        if prompt_text_override is not None:
            row_components = (prompt_text_override, row_components[1], row_components[2])
        for segment_id, text in enumerate(row_components):
            part = tokenizer(text, add_special_tokens=False)["input_ids"]
            ids.extend(part)
            segments.extend([segment_id] * len(part))
        ids.append(eos_id)
        segments.append(-1)
        original_count = len(ids)
        ids, segments = ids[:max_length], segments[:max_length]
        valid = [1] * len(ids)
        sequences.append((ids, segments, valid))
        metadata.append({
            "row_index": int(row["row_index"]),
            "document_id": row.get("document_id"),
            "article_id": row.get("article_id"),
            "variant": variant,
            "prompt_condition": "no_prompt_natural" if variant == "plain" else "task_prompt",
            "original_token_count": original_count,
            "saved_token_count": len(ids),
            "truncated": original_count > max_length,
            "prompt_token_slots": prompt_length,
            **sequence_audit(ids, segments, valid),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_size = int(model.config.hidden_size)
    prompt_path = output_dir / "prompt_token_embeddings.npy"
    prompt_matrix = None
    if prompt_length:
        prompt_matrix = np.lib.format.open_memmap(
            prompt_path, mode="w+", dtype=np.float32,
            shape=(len(rows), prompt_length, hidden_size),
        )
        np.save(output_dir / "prompt_input_ids.npy", np.asarray(prompt_ids, dtype=np.int32))
        (output_dir / "prompt_tokens.json").write_text(json.dumps({
            "text": prompt_text,
            "tokens": tokenizer.convert_ids_to_tokens(prompt_ids),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    names = (
        "prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean",
        "cls", "title_max", "body_max", "title_body_max", "full_max",
    )
    pooled = {name: [] for name in names}
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            width = max(len(item[0]) for item in batch)
            input_ids = np.full((len(batch), width), pad_id, dtype=np.int64)
            attention = np.zeros((len(batch), width), dtype=np.int64)
            segment_ids = np.full((len(batch), width), -2, dtype=np.int8)
            for index, (ids, segments, _) in enumerate(batch):
                input_ids[index, :len(ids)] = ids
                attention[index, :len(ids)] = 1
                segment_ids[index, :len(ids)] = segments
            attention_t = torch.from_numpy(attention).to(device)
            hidden = model(
                input_ids=torch.from_numpy(input_ids).to(device), attention_mask=attention_t,
            ).last_hidden_state
            segments_t = torch.from_numpy(segment_ids).to(device)
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
                "title_max": masked_max(hidden, masks["title"], torch),
                "body_max": masked_max(hidden, masks["body"], torch),
                "title_body_max": masked_max(hidden, masks["title_body"], torch),
                "full_max": masked_max(hidden, masks["full"], torch),
            }
            for name, value in values.items():
                pooled[name].append(value.cpu().numpy().astype(np.float32))
            if prompt_matrix is not None:
                prompt_hidden = hidden[:, 1:1 + prompt_length, :]
                if tuple(prompt_hidden.shape[1:]) != (prompt_length, hidden_size):
                    raise RuntimeError(f"unexpected prompt shape {tuple(prompt_hidden.shape)}")
                prompt_matrix[start:start + len(batch)] = prompt_hidden.cpu().numpy().astype(np.float32)
                prompt_matrix.flush()
            del hidden, values
    if prompt_matrix is not None:
        del prompt_matrix

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in pooled.items()}
    if variant == "plain":
        arrays.pop("prompt_mean")
        arrays.pop("title_mean")
        arrays.pop("title_max")
    np.savez_compressed(output_dir / "short_pooling.npz", **arrays)
    pd.DataFrame(metadata).to_json(output_dir / "metadata.jsonl", orient="records", lines=True, force_ascii=False)
    outputs = {name: list(value.shape) for name, value in arrays.items()}
    if prompt_length:
        outputs["prompt_token_embeddings"] = [len(rows), prompt_length, hidden_size]
    summary = {
        "format_version": "sina_cninfo_method_v1",
        "model": None,
        "variant": variant,
        "prompt_condition": "no_prompt_natural" if variant == "plain" else "task_prompt",
        "rows": len(rows), "max_length": max_length,
        "prompt_token_slots": prompt_length,
        "hidden_size": hidden_size,
        "outputs": outputs,
        "prompt_tokens_streamed_to_memmap": bool(prompt_length),
        "full_sequence_hidden_states_retained": False,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_PATHS), required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    rows = list(iter_records(args.input, None))
    if not rows:
        raise ValueError("input contains no rows")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(MODEL_PATHS[args.model])
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model missing: {model_path}")
    max_length = args.max_length or DEFAULT_MAX_LENGTH[args.model]
    batch_size = args.batch_size or (2 if args.model == "xlm_roberta_large" else 4)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    args.output_dir.mkdir(parents=True)
    try:
        for variant in VARIANTS:
            summary = encode_variant(
                rows, model=model, tokenizer=tokenizer, torch=torch, device=device,
                variant=variant, output_dir=args.output_dir / variant,
                batch_size=batch_size, max_length=max_length,
            )
            summary["model"] = args.model
            summary["model_path"] = str(model_path)
            (args.output_dir / variant / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps({"model": args.model, "variant": variant, **summary["outputs"]}, ensure_ascii=False), flush=True)
    except Exception:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
