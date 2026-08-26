"""Save every correctly aligned prompt-token hidden state with bounded memory.

Unlike the legacy pooled writer, this script writes directly to ``.npy``
memmaps and extracts prompt positions after BOS as ``[1:1+prompt_length]``.
Full-sequence hidden states are released after every batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import (
    DEFAULT_MAX_LENGTH,
    MODEL_PATHS,
    components,
    iter_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_PATHS), required=True)
    parser.add_argument("--variant", choices=("short", "masked_short", "long", "masked_long"), required=True)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--prompt-spec", type=Path)
    parser.add_argument("--prompt-key", choices=("short", "long"))
    args = parser.parse_args()
    max_length = args.max_length or DEFAULT_MAX_LENGTH[args.model]
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    import torch
    from transformers import AutoModel, AutoTokenizer

    if (args.prompt_spec is None) != (args.prompt_key is None):
        raise ValueError("--prompt-spec and --prompt-key must be provided together")
    prompt_override = None
    if args.prompt_spec is not None:
        specification = json.loads(args.prompt_spec.read_text(encoding="utf-8"))
        try:
            prompt_override = str(specification["prompts"][args.prompt_key]["text"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid prompt specification: {args.prompt_spec}") from exc
        expected_key = "long" if "long" in args.variant else "short"
        if args.prompt_key != expected_key:
            raise ValueError(
                f"variant {args.variant} requires prompt key {expected_key}, got {args.prompt_key}"
            )
        if not prompt_override:
            raise ValueError("prompt override must not be empty")

    rows = list(iter_records(args.input, args.rows))
    if not rows:
        raise ValueError("input contains no records")
    model_path = MODEL_PATHS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    empty_ids = tokenizer("", add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if len(empty_ids) < 2 or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must provide BOS/EOS and PAD tokens")
    bos_id, eos_id, pad_id = int(empty_ids[0]), int(empty_ids[-1]), int(tokenizer.pad_token_id)

    first_prompt = prompt_override or components(rows[0], args.variant)[0]
    prompt_ids = np.asarray(
        tokenizer(first_prompt, add_special_tokens=False)["input_ids"], dtype=np.int32
    )
    prompt_length = int(len(prompt_ids))
    if prompt_length < 1 or prompt_length + 2 > max_length:
        raise ValueError(f"invalid prompt length {prompt_length} for max_length={max_length}")
    hidden_size = int(model.config.hidden_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = args.output_dir / "prompt_token_embeddings.npy"
    matrix = np.lib.format.open_memmap(
        embedding_path, mode="w+", dtype=np.float32,
        shape=(len(rows), prompt_length, hidden_size),
    )
    np.save(args.output_dir / "prompt_input_ids.npy", prompt_ids)
    tokens = tokenizer.convert_ids_to_tokens(prompt_ids.tolist())
    (args.output_dir / "prompt_tokens.json").write_text(
        json.dumps({"text": first_prompt, "tokens": tokens}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata: list[dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            sequences: list[list[int]] = []
            original_counts: list[int] = []
            for row in batch_rows:
                prompt, title, body = components(row, args.variant)
                if prompt_override is not None:
                    prompt = prompt_override
                ids_prompt = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                if ids_prompt != prompt_ids.tolist():
                    raise ValueError(f"prompt token IDs changed at row_index={row.get('row_index')}")
                ids_title = tokenizer(title, add_special_tokens=False)["input_ids"]
                ids_body = tokenizer(body, add_special_tokens=False)["input_ids"]
                sequence = [bos_id] + ids_prompt + ids_title + ids_body + [eos_id]
                original_counts.append(len(sequence))
                sequences.append(sequence[:max_length])
            width = max(len(sequence) for sequence in sequences)
            input_ids = np.full((len(sequences), width), pad_id, dtype=np.int64)
            attention = np.zeros((len(sequences), width), dtype=np.int64)
            for index, sequence in enumerate(sequences):
                input_ids[index, :len(sequence)] = sequence
                attention[index, :len(sequence)] = 1
            hidden = model(
                input_ids=torch.from_numpy(input_ids).to(device),
                attention_mask=torch.from_numpy(attention).to(device),
            ).last_hidden_state
            # Position 0 is BOS. Preserve every prompt token, in order.
            prompt_hidden = hidden[:, 1:1 + prompt_length, :]
            if prompt_hidden.shape[1:] != (prompt_length, hidden_size):
                raise RuntimeError(f"unexpected prompt hidden shape: {tuple(prompt_hidden.shape)}")
            matrix[start:start + len(batch_rows)] = prompt_hidden.cpu().numpy().astype(np.float32)
            for offset, (row, original_count) in enumerate(zip(batch_rows, original_counts)):
                metadata.append({
                    "row_index": row.get("row_index"),
                    "document_id": row.get("document_id"),
                    "variant": args.variant,
                    "output_row": start + offset,
                    "prompt_token_count": prompt_length,
                    "original_token_count": original_count,
                    "saved_sequence_token_count": min(original_count, max_length),
                    "sequence_truncated": original_count > max_length,
                    "prompt_offset_start": 1,
                    "prompt_offset_end_exclusive": 1 + prompt_length,
                })
            del hidden, prompt_hidden
            matrix.flush()

    del matrix
    metadata_path = args.output_dir / "metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metadata),
        encoding="utf-8",
    )
    output_bytes = embedding_path.stat().st_size
    summary = {
        "format_version": "prompt_tokens_v5_bos_offset_fixed",
        "input": str(args.input),
        "input_parts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(args.input.rglob("part-*.jsonl"))
        ] if args.input.is_dir() else [{
            "path": str(args.input),
            "bytes": args.input.stat().st_size,
        }],
        "model": args.model,
        "model_path": model_path,
        "variant": args.variant,
        "rows": len(rows),
        "max_length": max_length,
        "prompt_token_count": prompt_length,
        "hidden_size": hidden_size,
        "shape": [len(rows), prompt_length, hidden_size],
        "dtype": "float32",
        "bytes": output_bytes,
        "gib": output_bytes / 1024**3,
        "prompt_slice": f"[1:{1 + prompt_length}] after BOS",
        "prompt_spec": str(args.prompt_spec) if args.prompt_spec else None,
        "prompt_key": args.prompt_key,
        "prompt_sha256": hashlib.sha256(first_prompt.encode("utf-8")).hexdigest(),
        "embedding_file": str(embedding_path),
        "prompt_input_ids_file": str(args.output_dir / "prompt_input_ids.npy"),
        "metadata_file": str(metadata_path),
        "full_sequence_hidden_states_retained": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
