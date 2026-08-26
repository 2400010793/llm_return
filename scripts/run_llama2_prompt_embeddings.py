"""Generate causal Llama-2 readout and pooled embeddings for one prompt shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import components, iter_records


POOLING_NAMES = (
    "prompt_mean",
    "title_mean",
    "body_mean",
    "title_body_mean",
    "full_mean",
    "cls",
    "title_max",
    "body_max",
    "title_body_max",
    "full_max",
)


def compose_causal_readout_sequence(
    title_ids: list[int],
    body_ids: list[int],
    separator_ids: list[int],
    prompt_ids: list[int],
    *,
    bos_id: int | None,
    eos_id: int,
    max_length: int,
) -> tuple[list[int], list[int], dict[str, int | bool]]:
    """Place the readout prompt last so its states can attend to the article."""
    boundary_count = 1 + int(bos_id is not None)
    article_budget = (
        max_length - len(separator_ids) - len(prompt_ids) - boundary_count
    )
    if article_budget < 1:
        raise ValueError(
            "max_length must leave at least one article token before the readout prompt"
        )
    article_ids = (title_ids + body_ids)[:article_budget]
    visible_title = min(len(title_ids), article_budget)
    visible_body = len(article_ids) - visible_title
    prefix = [] if bos_id is None else [bos_id]
    ids = [*prefix, *article_ids, *separator_ids, *prompt_ids, eos_id]
    segments = [
        *([-1] * len(prefix)),
        *([1] * visible_title),
        *([2] * visible_body),
        *([-1] * len(separator_ids)),
        *([0] * len(prompt_ids)),
        -1,
    ]
    boundary_count = len(prefix) + 1
    original_tokens = boundary_count + len(title_ids) + len(body_ids) + len(separator_ids) + len(prompt_ids)
    return ids, segments, {
        "original_token_count": original_tokens,
        "saved_token_count": len(ids),
        "sequence_truncated": original_tokens > len(ids),
        "visible_title_tokens": visible_title,
        "visible_body_tokens": visible_body,
        "separator_token_count": len(separator_ids),
        "prompt_start_zero_based": len(prefix) + len(article_ids) + len(separator_ids),
        "prompt_end_zero_based": len(prefix) + len(article_ids) + len(separator_ids) + len(prompt_ids) - 1,
    }


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


def masked_max(hidden, mask, torch):
    values = hidden.masked_fill(
        ~mask.unsqueeze(-1), torch.finfo(hidden.dtype).min,
    ).max(1).values
    empty = ~mask.any(1)
    if empty.any():
        values[empty] = 0
    return values


def _storage_dtype(name: str) -> np.dtype:
    return np.dtype({"float16": np.float16, "float32": np.float32}[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--variant", choices=("short", "masked_short"), required=True)
    parser.add_argument("--prompt-spec", type=Path, required=True)
    parser.add_argument("--prompt-key", default="short")
    parser.add_argument("--model-name", default="llama2_13b")
    parser.add_argument("--format-version", default="llama2_causal_readout_v2")
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--flush-every", type=int, default=256)
    args = parser.parse_args()
    if min(args.batch_size, args.max_length, args.flush_every) < 1:
        raise ValueError("batch-size, max-length, and flush-every must be positive")
    if not args.input.is_dir():
        raise FileNotFoundError(f"input shard is unavailable: {args.input}")
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"model is unavailable: {args.model_path}")

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    torch.cuda.reset_peak_memory_stats()
    specification = json.loads(args.prompt_spec.read_text(encoding="utf-8"))
    prompt = str(specification["prompts"][args.prompt_key]["text"])
    rows = list(iter_records(args.input, args.rows))
    if not rows:
        raise ValueError("input contains no rows")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model_path, local_files_only=True, dtype=torch.float16,
    ).to("cuda").eval()
    if tokenizer.eos_token_id is None:
        raise ValueError("causal tokenizer lacks an EOS boundary token")
    bos_id = (
        int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None
    )
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    separator_ids = tokenizer(args.separator, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        raise ValueError("readout prompt tokenized to an empty sequence")
    hidden_size = int(model.config.hidden_size)
    storage_dtype = _storage_dtype(args.storage_dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_shape = (len(rows), len(prompt_ids), hidden_size)
    prompt_memmap = np.lib.format.open_memmap(
        args.output_dir / "prompt_token_embeddings.npy",
        mode="w+", dtype=storage_dtype, shape=prompt_shape,
    )
    np.save(
        args.output_dir / "prompt_input_ids.npy",
        np.asarray(prompt_ids, dtype=np.int32),
    )
    (args.output_dir / "prompt_tokens.json").write_text(
        json.dumps({
            "text": prompt,
            "tokens": tokenizer.convert_ids_to_tokens(prompt_ids),
            "layout": "article_then_readout_prompt",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in POOLING_NAMES}
    metadata_path = args.output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_handle:
        with torch.inference_mode():
            for batch_number, start in enumerate(range(0, len(rows), args.batch_size), 1):
                batch = rows[start:start + args.batch_size]
                sequences: list[list[int]] = []
                segments: list[list[int]] = []
                audits: list[dict[str, int | bool]] = []
                for row in batch:
                    _, title, body = components(row, args.variant)
                    title_ids = tokenizer(title, add_special_tokens=False)["input_ids"]
                    body_ids = tokenizer(body, add_special_tokens=False)["input_ids"]
                    ids, segment_ids, sequence_audit = compose_causal_readout_sequence(
                        title_ids, body_ids, separator_ids, prompt_ids,
                        bos_id=bos_id, eos_id=eos_id, max_length=args.max_length,
                    )
                    sequences.append(ids)
                    segments.append(segment_ids)
                    audits.append(sequence_audit)

                width = max(map(len, sequences))
                ids_np = np.full((len(batch), width), pad_id, dtype=np.int64)
                attention_np = np.zeros((len(batch), width), dtype=np.int64)
                segment_np = np.full((len(batch), width), -2, dtype=np.int8)
                for index, (ids, segment_ids) in enumerate(zip(sequences, segments)):
                    ids_np[index, :len(ids)] = ids
                    attention_np[index, :len(ids)] = 1
                    segment_np[index, :len(segment_ids)] = segment_ids

                attention = torch.from_numpy(attention_np).to("cuda")
                hidden = model(
                    input_ids=torch.from_numpy(ids_np).to("cuda"),
                    attention_mask=attention,
                ).last_hidden_state
                segment_tensor = torch.from_numpy(segment_np).to("cuda")
                visible = attention.bool()
                masks = {
                    "prompt": (segment_tensor == 0) & visible,
                    "title": (segment_tensor == 1) & visible,
                    "body": (segment_tensor == 2) & visible,
                }
                masks["title_body"] = masks["title"] | masks["body"]
                masks["full"] = masks["prompt"] | masks["title_body"]
                prompt_hidden = torch.stack([
                    hidden[index][masks["prompt"][index]]
                    for index in range(len(batch))
                ])
                if prompt_hidden.shape[1] != len(prompt_ids):
                    raise ValueError("readout prompt was not fully preserved after truncation")
                values = {
                    "prompt_mean": prompt_hidden.mean(1),
                    "title_mean": masked_mean(hidden, masks["title"]),
                    "body_mean": masked_mean(hidden, masks["body"]),
                    "title_body_mean": masked_mean(hidden, masks["title_body"]),
                    "full_mean": masked_mean(hidden, masks["full"]),
                    # Generic downstream code calls this feature cls.  For a
                    # causal model, the final readout token is its analogue.
                    "cls": prompt_hidden[:, -1, :],
                    "title_max": masked_max(hidden, masks["title"], torch),
                    "body_max": masked_max(hidden, masks["body"], torch),
                    "title_body_max": masked_max(hidden, masks["title_body"], torch),
                    "full_max": masked_max(hidden, masks["full"], torch),
                }
                prompt_values = prompt_hidden.detach().cpu().numpy().astype(
                    storage_dtype, copy=False,
                )
                prompt_memmap[start:start + len(batch)] = prompt_values
                for name, value in values.items():
                    pooled[name].append(
                        value.float().cpu().numpy().astype(storage_dtype, copy=False)
                    )
                for row, sequence_audit in zip(batch, audits):
                    metadata_handle.write(json.dumps({
                        "row_index": row.get("row_index"),
                        "document_id": row.get("document_id"),
                        "article_id": row.get("article_id"),
                        "variant": args.variant,
                        "sequence_layout": "article_then_readout_prompt",
                        **sequence_audit,
                    }, ensure_ascii=False) + "\n")
                if batch_number % args.flush_every == 0:
                    prompt_memmap.flush()
                    metadata_handle.flush()
                del hidden, values, prompt_hidden, prompt_values

    prompt_memmap.flush()
    del prompt_memmap
    arrays = {
        name: np.concatenate(parts).astype(storage_dtype, copy=False)
        for name, parts in pooled.items()
    }
    np.savez_compressed(args.output_dir / "short_pooling.npz", **arrays)
    summary = {
        "format_version": args.format_version,
        "model": args.model_name,
        "model_path": str(args.model_path.resolve()),
        "variant": args.variant,
        "rows": len(rows),
        "max_length": args.max_length,
        "sequence_layout": "article_then_readout_prompt",
        "separator": args.separator,
        "separator_token_ids": separator_ids,
        "prompt_token_count": len(prompt_ids),
        "hidden_size": hidden_size,
        "storage_dtype": args.storage_dtype,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "outputs": {name: list(values.shape) for name, values in arrays.items()},
        "prompt_token_embeddings": list(prompt_shape),
        "cls_pooling": "last_readout_token",
        "cuda_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        ),
        "cuda_peak_reserved_mib": float(
            torch.cuda.max_memory_reserved() / (1024 * 1024)
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
