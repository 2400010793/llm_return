"""Generate all clean aligned prompts for one masked-short RoBERTa shard."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import MODEL_PATHS, iter_records
from scripts.run_sina_cninfo_method_embeddings import encode_variant
from src.data.aligned_prompts import audit_aligned_prompts, load_aligned_prompt_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta",), default="roberta")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=None, help="optional bounded probe size")
    parser.add_argument("--group-id", type=int, default=None, help="encode only one configured prompt group")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    config = load_aligned_prompt_config(args.config)
    if config.get("model") != args.model:
        raise ValueError(f"config model {config.get('model')} does not match {args.model}")
    rows = list(iter_records(args.input, args.rows))
    if not rows:
        raise ValueError("input contains no rows")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(MODEL_PATHS[args.model])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    audit = audit_aligned_prompts(config, tokenizer)
    group_map = config.get("task_groups", {})
    if args.group_id is None:
        selected_axes = list(config["axes"])
    else:
        selected_axes = group_map.get(str(args.group_id))
        if not selected_axes:
            raise ValueError(f"unknown or empty prompt group: {args.group_id}")
        unknown = set(selected_axes).difference(config["axes"])
        if unknown:
            raise ValueError(f"prompt group references unknown axes: {sorted(unknown)}")
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for axis in selected_axes:
            spec = config["axes"][axis]
            for level in ("low", "neutral", "high"):
                item = spec["levels"][level]
                prompt_id = str(item["prompt_id"])
                out = args.output_dir / prompt_id / "masked_short"
                summary = encode_variant(
                    rows, model=model, tokenizer=tokenizer, torch=torch, device=device,
                    variant="masked_short", output_dir=out, batch_size=args.batch_size,
                    max_length=args.max_length, prompt_text_override=str(item["text"]),
                )
                expected_outputs = {
                    "prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean",
                    "cls", "title_max", "body_max", "title_body_max", "full_max",
                    "prompt_token_embeddings",
                }
                actual_outputs = set(summary.get("outputs", {}))
                if actual_outputs != expected_outputs:
                    raise RuntimeError(
                        f"RoBERTa pooled output contract changed for {prompt_id}: "
                        f"expected={sorted(expected_outputs)} actual={sorted(actual_outputs)}"
                    )
                prompt_audit = audit["prompts"][prompt_id]
                prompt_spec = {
                    "format_version": "aligned_masked_short_prompt_v1",
                    "axis": axis,
                    "level": level,
                    "prompt_id": prompt_id,
                    "prompt_text": item["text"],
                    "target": item["target"],
                    "variant": "masked_short",
                    "mask_policy": "identity_and_time_only",
                    "model": args.model,
                    "tokenizer_sha256": audit["tokenizer_sha256"],
                    "token_audit": prompt_audit,
                    "summary": summary,
                }
                (out / "prompt_spec.json").write_text(
                    json.dumps(prompt_spec, ensure_ascii=False, indent=2), encoding="utf-8",
                )
        manifest = {
            "format_version": "aligned_masked_short_embeddings_v1",
            "model": args.model,
            "variant": "masked_short",
            "rows": len(rows),
            "max_length": args.max_length,
            "config": str(args.config),
            "tokenizer_sha256": audit["tokenizer_sha256"],
            "axes": list(config["axes"]),
            "selected_axes": selected_axes,
            "group_id": args.group_id,
            "prompt_count": sum(3 for _ in selected_axes),
            "mask_policy": "identity_and_time_only",
            "pooled_representations": [
                "prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean",
                "cls", "title_max", "body_max", "title_body_max", "full_max",
            ],
            "token_representation": "prompt_token_embeddings",
        }
        (args.output_dir / "preflight.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (args.output_dir / "COMPLETED").write_text("aligned_masked_short_embeddings_v1\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
