"""Generate short/masked-short prompt embeddings for several fixed templates.

This is a thin orchestration layer over the existing Sina/CNINFO embedding
encoder. The Transformer is loaded once per shard/model; only prompt text and
output directories vary.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import MODEL_PATHS, DEFAULT_MAX_LENGTH, iter_records
from scripts.run_sina_cninfo_method_embeddings import encode_variant


PROMPTS = {
    "profit": "分析股票盈利",
    "excess_return": "分析股票超额收益",
    "return": "分析股票收益",
    "loss": "分析股票亏损",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", choices=tuple(MODEL_PATHS), required=True)
    p.add_argument("--prompt-config", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=None)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = list(iter_records(args.input, None))
    if not rows:
        raise ValueError("input contains no rows")
    config = json.loads(args.prompt_config.read_text(encoding="utf-8"))
    prompts = config.get("prompts", PROMPTS)
    if set(prompts) != set(PROMPTS):
        raise ValueError(f"prompt config must contain exactly {sorted(PROMPTS)}")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(MODEL_PATHS[args.model])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    max_length = args.max_length or DEFAULT_MAX_LENGTH[args.model]
    try:
        for prompt_id, prompt_text in prompts.items():
            for variant in ("short", "masked_short"):
                out = args.output_dir / prompt_id / variant
                encode_variant(
                    rows, model=model, tokenizer=tokenizer, torch=torch, device=device,
                    variant=variant, output_dir=out, batch_size=args.batch_size,
                    max_length=max_length, prompt_text_override=prompt_text,
                )
                (out / "prompt_spec.json").write_text(json.dumps({
                    "prompt_id": prompt_id, "prompt_text": prompt_text,
                    "model": args.model, "variant": variant, "rows": len(rows),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output_dir / "COMPLETED").write_text("prompt_template_embeddings_v1\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()