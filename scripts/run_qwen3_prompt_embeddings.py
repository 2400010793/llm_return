"""Extract Qwen3-Embedding-8B token readouts for one news shard.

The article is encoded before a fixed separator and prompt.  The prompt is
therefore at the end of the causal sequence and can attend to the article.
This entry point requires a Transformers-format local model directory; an
Ollama pooled-embedding endpoint cannot provide ``last_hidden_state``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from scripts.run_short_pooled_embeddings import components, iter_records


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _compose(article, separator, prompt, eos, max_length):
    budget = max_length - len(separator) - len(prompt) - (1 if eos is not None else 0)
    if budget < 1:
        raise ValueError("max_length leaves no room for article tokens")
    article = article[:budget]
    ids = article + separator + prompt
    if eos is not None:
        ids.append(eos)
    prompt_start = len(article) + len(separator)
    return ids, prompt_start, len(article)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--variant", choices=("short", "masked_short"), required=True)
    p.add_argument("--separator", default="\n\n")
    p.add_argument("--max-length", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--rows", type=int)
    args = p.parse_args()
    if not args.model_path.joinpath("config.json").is_file():
        raise FileNotFoundError(f"Transformers model directory missing: {args.model_path}")
    rows = list(iter_records(args.input, args.rows))
    if not rows:
        raise ValueError("input shard contains no rows")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    eos = tokenizer.eos_token_id
    prompt_ids = _encode(tokenizer, args.prompt)
    separator_ids = _encode(tokenizer, args.separator)
    if not prompt_ids:
        raise ValueError("prompt tokenized to empty sequence")
    hidden_size = int(model.config.hidden_size)
    n = len(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_out = np.lib.format.open_memmap(
        args.output_dir / "prompt_token_embeddings.npy", mode="w+",
        dtype=np.float16, shape=(n, len(prompt_ids), hidden_size),
    )
    last_out = np.lib.format.open_memmap(
        args.output_dir / "last_token_embeddings.npy", mode="w+",
        dtype=np.float16, shape=(n, hidden_size),
    )
    np.save(args.output_dir / "prompt_input_ids.npy", np.asarray(prompt_ids, dtype=np.int32))
    (args.output_dir / "prompt_tokens.json").write_text(json.dumps({
        "text": args.prompt, "tokens": tokenizer.convert_ids_to_tokens(prompt_ids),
        "layout": "article_separator_prompt_eos", "max_length": args.max_length,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = []
    for start in range(0, n, args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        composed = []
        for row in batch_rows:
            prompt_text, title, body = components(row, args.variant)
            article = _encode(tokenizer, title + body)
            ids, prompt_start, article_tokens = _compose(
                article, separator_ids, prompt_ids, eos, args.max_length,
            )
            composed.append((ids, prompt_start, article_tokens, len(article)))
        width = max(len(item[0]) for item in composed)
        input_ids = np.full((len(composed), width), tokenizer.pad_token_id, dtype=np.int64)
        attention = np.zeros_like(input_ids)
        for i, (ids, _, _, _) in enumerate(composed):
            input_ids[i, :len(ids)] = ids
            attention[i, :len(ids)] = 1
        with torch.inference_mode():
            output = model(
                input_ids=torch.from_numpy(input_ids).cuda(),
                attention_mask=torch.from_numpy(attention).cuda(),
            )
            hidden = output.last_hidden_state
        for i, (ids, prompt_start, article_tokens, original_article) in enumerate(composed):
            length = len(ids)
            prompt_out[start + i] = hidden[i, prompt_start:prompt_start + len(prompt_ids)].float().cpu().numpy()
            last_out[start + i] = hidden[i, length - 1].float().cpu().numpy()
            metadata.append({
                "row_index": int(batch_rows[i]["row_index"]),
                "variant": args.variant,
                "original_article_tokens": original_article,
                "visible_article_tokens": article_tokens,
                "saved_token_count": length,
                "truncated": original_article > article_tokens,
                "prompt_start_zero_based": prompt_start,
                "prompt_token_count": len(prompt_ids),
                "separator_token_count": len(separator_ids),
            })
        del output, hidden
        torch.cuda.empty_cache()
    prompt_out.flush(); last_out.flush()
    (args.output_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(json.dumps({
        "model": str(args.model_path), "rows": n, "hidden_size": hidden_size,
        "max_length": args.max_length, "variant": args.variant,
        "prompt": args.prompt, "prompt_tokens": len(prompt_ids),
        "separator_tokens": len(separator_ids), "storage_dtype": "float16",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()