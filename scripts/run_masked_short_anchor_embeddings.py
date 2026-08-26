"""Generate masked-short RoBERTa embeddings for no-neutral-marker anchors."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_short_pooled_embeddings import MODEL_PATHS, iter_records
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
    parser.add_argument("--model", choices=("roberta",), default="roberta")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("variant") != "masked_short" or config.get("model") != args.model:
        raise ValueError("anchor config must target masked_short RoBERTa")
    rows = list(iter_records(args.input, args.rows))
    if not rows:
        raise ValueError("input contains no rows")
    axes = config.get("task_groups", {}).get(str(args.group_id))
    prompts = config.get("prompts", {})
    if not axes or any(axis not in prompts for axis in axes):
        raise ValueError(f"invalid anchor group {args.group_id}: {axes}")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(MODEL_PATHS[args.model])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_hash = tokenizer_digest(tokenizer)
    try:
        audits = {}
        for axis in axes:
            item = prompts[axis]
            text = str(item["text"])
            target = str(item["target"])
            if "中" in text or text.count(target) != 1:
                raise ValueError(f"invalid no-neutral anchor: {text} / {target}")
            prompt_id = str(item["prompt_id"])
            audits[prompt_id] = _prompt_audit(text, target, tokenizer)
            out = args.output_dir / prompt_id / "masked_short"
            summary = encode_variant(
                rows, model=model, tokenizer=tokenizer, torch=torch, device=device,
                variant="masked_short", output_dir=out, batch_size=args.batch_size,
                max_length=args.max_length, prompt_text_override=text,
            )
            actual = set(summary.get("outputs", {}))
            if actual != EXPECTED_OUTPUTS:
                raise RuntimeError(f"RoBERTa output contract changed: {sorted(actual)}")
            (out / "prompt_spec.json").write_text(json.dumps({
                "format_version": "masked_short_anchor_prompt_v1",
                "axis": axis, "level": "anchor_without_neutral_marker",
                "prompt_id": prompt_id, "prompt_text": text, "target": target,
                "variant": "masked_short", "mask_policy": "identity_and_time_only",
                "model": args.model, "tokenizer_sha256": tokenizer_hash,
                "token_audit": audits[prompt_id], "summary": summary,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {
            "format_version": "masked_short_anchor_embeddings_v1",
            "model": args.model, "variant": "masked_short", "rows": len(rows),
            "max_length": args.max_length, "group_id": args.group_id,
            "selected_axes": axes, "prompt_count": len(axes),
            "anchor_rule": "prompt text contains no 中 marker",
            "tokenizer_sha256": tokenizer_hash, "prompts": audits,
        }
        (args.output_dir / "preflight.json").write_text(json.dumps({
            "format_version": "masked_short_anchor_preflight_v1", "prompts": audits
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output_dir / "COMPLETED").write_text("masked_short_anchor_embeddings_v1\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
