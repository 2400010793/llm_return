"""Extract only末尾 prompt tokens and article-mean vectors from local Qwen3 GGUF."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from llama_cpp import Llama
from llama_cpp import llama_cpp

from scripts.run_short_pooled_embeddings import components, iter_records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--variant", choices=("short", "masked_short"), required=True)
    p.add_argument("--max-length", type=int, default=500)
    p.add_argument("--rows", type=int)
    args = p.parse_args()
    if args.output_dir.joinpath("COMPLETED").is_file():
        return
    rows = list(iter_records(args.input, args.rows))
    if not rows:
        raise ValueError("empty input shard")
    stage = args.output_dir.with_name(f".{args.output_dir.name}.partial.{os.getpid()}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        model = Llama(
            model_path=str(args.model), n_ctx=args.max_length,
            n_batch=args.max_length, n_gpu_layers=-1, embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE, verbose=False,
        )
        prompt_ids = model.tokenize(args.prompt.encode(), add_bos=False, special=False)
        sep_ids = model.tokenize(b"\n\n", add_bos=False, special=False)
        eos = model.token_eos()
        hidden_size = model.n_embd()
        prompt_out = []
        article_out = []
        metadata = []
        for row in rows:
            _, title, body = components(row, args.variant)
            article_ids = model.tokenize((title + body).encode(), add_bos=False, special=False)
            budget = args.max_length - len(sep_ids) - len(prompt_ids) - 1
            visible_article = article_ids[:budget]
            ids = visible_article + sep_ids + prompt_ids + [eos]
            model.reset()
            model.eval(ids)
            ptr = llama_cpp.llama_get_embeddings(model._ctx.ctx)
            values = np.ctypeslib.as_array(
                ptr, shape=(len(ids) * hidden_size,)
            ).reshape(len(ids), hidden_size).copy()
            prompt_start = len(visible_article) + len(sep_ids)
            prompt_out.append(values[prompt_start:prompt_start + len(prompt_ids)])
            article_out.append(values[:len(visible_article)].mean(axis=0, dtype=np.float32))
            metadata.append({
                "row_index": int(row["row_index"]), "variant": args.variant,
                "original_article_tokens": len(article_ids),
                "visible_article_tokens": len(visible_article),
                "saved_token_count": len(ids),
                "truncated": len(article_ids) > len(visible_article),
                "prompt_start_zero_based": prompt_start,
                "prompt_token_count": len(prompt_ids),
                "separator_token_count": len(sep_ids),
            })
        np.save(stage / "prompt_token_embeddings.npy", np.asarray(prompt_out, dtype=np.float16))
        np.save(stage / "article_mean_embeddings.npy", np.asarray(article_out, dtype=np.float16))
        np.save(stage / "prompt_input_ids.npy", np.asarray(prompt_ids, dtype=np.int32))
        (stage / "prompt_tokens.json").write_text(json.dumps({
            "text": args.prompt, "tokens": [model.detokenize([x]).decode("utf-8", "replace") for x in prompt_ids],
            "layout": "article_separator_prompt_eos", "max_length": args.max_length,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (stage / "metadata.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in metadata), encoding="utf-8",
        )
        (stage / "summary.json").write_text(json.dumps({
            "model": str(args.model), "rows": len(rows), "hidden_size": hidden_size,
            "prompt": args.prompt, "variant": args.variant, "max_length": args.max_length,
            "outputs": ["prompt_token_embeddings", "article_mean_embeddings"],
            "dtype": "float16", "pooling": "none",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (stage / "COMPLETED").write_text("qwen3_gguf_prompt_embeddings_v1\n", encoding="utf-8")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()