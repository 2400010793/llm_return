"""Encode six neutral masked-short prompts for one shard and one model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import DEFAULT_MAX_LENGTH, MODEL_PATHS, iter_records
from scripts.run_sina_cninfo_method_embeddings import encode_variant
from src.data.aligned_prompts import _prompt_audit, tokenizer_digest


EXPECTED_OUTPUTS = {
    "prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean",
    "cls", "title_max", "body_max", "title_body_max", "full_max",
    "prompt_token_embeddings",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prompts = config.get("prompts")
    if config.get("variant") != "masked_short" or not isinstance(prompts, dict) or len(prompts) != 6:
        raise ValueError("neutral config must contain exactly six masked_short prompts")
    rows = list(iter_records(args.input, args.rows))
    if not rows:
        raise ValueError("input contains no rows")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(MODEL_PATHS[args.model])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    max_length = args.max_length or DEFAULT_MAX_LENGTH[args.model]
    batch_size = args.batch_size or (4 if args.model == "bge_m3" else 8)
    if args.model == "roberta" and max_length > 512:
        raise ValueError("RoBERTa max_length cannot exceed 512")
    if args.model == "bge_m3" and max_length > 1000:
        raise ValueError("BGE-M3 max_length cannot exceed 1000")

    tokenizer_hash = tokenizer_digest(tokenizer)
    audits = {}
    for prompt_id, item in prompts.items():
        text = str(item.get("text", ""))
        target = str(item.get("target", ""))
        if not text or not target or "高" in text or "低" in text or "中" in text:
            raise ValueError(f"invalid neutral prompt {prompt_id}: {text!r}")
        audits[prompt_id] = _prompt_audit(text, target, tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for prompt_id, item in prompts.items():
            text = str(item["text"])
            out = args.output_dir / prompt_id / "masked_short"
            summary = encode_variant(
                rows, model=model, tokenizer=tokenizer, torch=torch, device=device,
                variant="masked_short", output_dir=out, batch_size=batch_size,
                max_length=max_length, prompt_text_override=text,
            )
            if set(summary.get("outputs", {})) != EXPECTED_OUTPUTS:
                raise RuntimeError(f"unexpected output contract for {prompt_id}")
            spec = {
                "format_version": "neutral_masked_short_prompt_v1",
                "prompt_id": prompt_id, "label": item.get("label", prompt_id),
                "target": item["target"], "prompt_text": text,
                "variant": "masked_short", "mask_policy": config["mask_policy"],
                "model": args.model, "tokenizer_sha256": tokenizer_hash,
                "token_audit": audits[prompt_id], "summary": summary,
            }
            (out / "prompt_spec.json").write_text(
                json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        manifest = {
            "format_version": "neutral_masked_short_embeddings_v1",
            "model": args.model, "variant": "masked_short", "rows": len(rows),
            "max_length": max_length, "batch_size": batch_size,
            "config": str(args.config), "tokenizer_sha256": tokenizer_hash,
            "prompt_count": len(prompts), "prompt_ids": list(prompts),
            "mask_policy": config["mask_policy"],
            "pooled_representations": sorted(EXPECTED_OUTPUTS - {"prompt_token_embeddings"}),
            "token_representation": "prompt_token_embeddings",
            "prompt_audits": audits,
        }
        (args.output_dir / "preflight.json").write_text(
            json.dumps({"tokenizer_sha256": tokenizer_hash, "prompts": audits}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "COMPLETED").write_text("neutral_masked_short_embeddings_v1\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
