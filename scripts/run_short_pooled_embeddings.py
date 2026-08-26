"""Encode short prompts and save only the requested pooled representations.

For each row and variant, one forward pass saves every prompt-token hidden
state, plus mean embeddings for title, body, and the concatenated title+body.
No full sequence hidden states are retained after a batch is processed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

PAPER_HK_MODEL_ROOT = Path(os.environ.get(
    "PAPER_HK_MODEL_ROOT",
    "/home/gaozh/.cache/huggingface/paper_hk_models",
))
FINBERT_MODEL_ROOT = Path(os.environ.get(
    "FINBERT_MODEL_ROOT",
    "/mnt/lustre3/home/gaozh/llm_return/models",
))
MODEL_PATHS = {
    "roberta": "/home/gaozh/llm_return/models/chinese-roberta-wwm-ext",
    "bge_m3": "/home/gaozh/llm_return/models/bge-m3",
    # Paper-faithful China (HK) encoders from Table IA11 of Chen, Kelly,
    # and Xiu.  They are downloaded explicitly before Slurm arrays start so
    # that one transient network error cannot stall every embedding task.
    "ckip_bert": str(PAPER_HK_MODEL_ROOT / "ckiplab-bert-base-chinese"),
    "xlm_roberta_large": str(PAPER_HK_MODEL_ROOT / "xlm-roberta-large"),
    "finbert2_base": str(FINBERT_MODEL_ROOT / "finbert2-base"),
}
DEFAULT_MAX_LENGTH = {
    "roberta": 512,
    "bge_m3": 1000,
    "ckip_bert": 512,
    "xlm_roberta_large": 512,
    "finbert2_base": 512,
}


def save_npz_atomic(path: Path, outputs: dict[str, np.ndarray], attempts: int = 3) -> None:
    """Stage large compressed arrays locally, then atomically publish them."""
    scratch_root = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    if not scratch_root.is_dir():
        scratch_root = path.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=scratch_root, prefix=f".{path.stem}.", suffix=".npz", delete=False,
    ) as handle:
        scratch = Path(handle.name)
    target_tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with scratch.open("wb") as handle:
            np.savez_compressed(handle, **outputs)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(1, attempts + 1):
            try:
                shutil.copyfile(scratch, target_tmp)
                with target_tmp.open("rb") as handle:
                    os.fsync(handle.fileno())
                with zipfile.ZipFile(target_tmp) as archive:
                    expected = {f"{name}.npy" for name in outputs}
                    if set(archive.namelist()) != expected:
                        raise ValueError("staged NPZ member names do not match outputs")
                    for name in expected:
                        with archive.open(name) as member:
                            member.read(1)
                os.replace(target_tmp, path)
                return
            except (OSError, ValueError, zipfile.BadZipFile):
                target_tmp.unlink(missing_ok=True)
                if attempt == attempts:
                    raise
                time.sleep(attempt)
    finally:
        scratch.unlink(missing_ok=True)
        target_tmp.unlink(missing_ok=True)


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
    if variant == "plain":
        # The paper feeds raw article text without a task prompt.  Put the
        # complete article in the body segment so full_mean/body_mean are the
        # exact masked mean of all visible article tokens.
        return "", "", str(row.get("text_plain", "") or "")
    masked = variant in ("masked_short", "masked_long")
    long = variant in ("long", "masked_long")
    prefix = ("masked_" if masked else "") + ("long" if long else "short")
    full_key = {"short": "text_input_2_short", "masked_short": "text_input_5_masked_short",
                "long": "text_input_7_fixed_long", "masked_long": "text_input_8_fixed_masked_long"}[variant]
    full = str(row.get(full_key, "") or "")
    prompt = str(row.get(f"{prefix}_prompt", "") or "")
    title = str(row.get(f"{prefix}_title", "") or "")
    body = str(row.get(f"{prefix}_body", "") or "")
    if not prompt:
        marker = "文章："
        pos = full.find(marker)
        prompt = full[:pos + len(marker)] if pos >= 0 else full
    if not body:
        body = full[len(prompt):] if full.startswith(prompt) else full
    return prompt, title, body


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def masked_max(hidden, mask, torch):
    masked = hidden.masked_fill(~mask.unsqueeze(-1), torch.finfo(hidden.dtype).min)
    values = masked.max(dim=1).values
    empty = ~mask.any(dim=1)
    if empty.any():
        values[empty] = 0
    return values


def conditioned_prompt_ids(
    prompt_ids: list[int], *, prompt_slots: int, pad_id: int, condition: str,
) -> tuple[list[int], list[int]]:
    """Build a Prompt segment and attention mask for one ablation condition."""
    fitted = prompt_ids[:prompt_slots] + [pad_id] * max(0, prompt_slots - len(prompt_ids))
    if condition == "task_prompt":
        return fitted, [int(token_id != pad_id) for token_id in fitted]
    if condition == "no_prompt_position_matched":
        return [pad_id] * prompt_slots, [0] * prompt_slots
    if condition == "no_prompt_natural":
        return [], []
    raise ValueError(f"unknown prompt condition: {condition}")


def sequence_audit(ids: list[int], segments: list[int], valid: list[int]) -> dict[str, object]:
    """Return auditable token positions/counts for one truncated model input."""
    def segment_summary(segment_id: int) -> tuple[int, int | None, int | None]:
        positions = [
            position for position, (segment, visible) in enumerate(zip(segments, valid))
            if segment == segment_id and visible
        ]
        return (
            len(positions),
            positions[0] if positions else None,
            positions[-1] if positions else None,
        )

    prompt_count, prompt_start, prompt_end = segment_summary(0)
    title_count, title_start, title_end = segment_summary(1)
    body_count, body_start, body_end = segment_summary(2)
    digest = hashlib.sha256()
    digest.update(json.dumps(
        {"input_ids": ids, "attention_mask": valid},
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8"))
    return {
        "visible_prompt_tokens": prompt_count,
        "visible_title_tokens": title_count,
        "visible_body_tokens": body_count,
        "prompt_start_zero_based": prompt_start,
        "prompt_end_zero_based": prompt_end,
        "title_start_zero_based": title_start,
        "title_end_zero_based": title_end,
        "body_start_zero_based": body_start,
        "body_end_zero_based": body_end,
        "input_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_PATHS), required=True)
    parser.add_argument(
        "--variant",
        choices=("plain", "short", "masked_short", "long", "masked_long"),
        required=True,
    )
    parser.add_argument(
        "--prompt-condition",
        choices=("task_prompt", "no_prompt_position_matched", "no_prompt_natural"),
        default="task_prompt",
    )
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--prompt-length", type=int, default=None)
    args = parser.parse_args()
    max_length = args.max_length or DEFAULT_MAX_LENGTH[args.model]
    if args.model in {
        "roberta", "ckip_bert", "xlm_roberta_large", "finbert2_base",
    } and max_length > 512:
        raise ValueError(f"{args.model} model limit is 512 tokens")
    if args.model == "bge_m3" and max_length > 1000:
        raise ValueError("This experiment caps BGE-M3 at 1000 tokens")
    if args.variant == "plain" and args.prompt_condition != "no_prompt_natural":
        raise ValueError("plain paper replication requires --prompt-condition no_prompt_natural")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = MODEL_PATHS[args.model]
    if not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"model is not available locally: {model_path}; run the paper-HK prefetch job first"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model_kwargs = {"add_pooling_layer": False} if args.model == "finbert2_base" else {}
    model = AutoModel.from_pretrained(
        model_path, local_files_only=True, **model_kwargs
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    special_ids = tokenizer("", add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if len(special_ids) < 2 or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must provide start/end special tokens and pad token")
    bos_id, eos_id, pad_id = special_ids[0], special_ids[-1], tokenizer.pad_token_id

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
            enc = tokenizer(text, add_special_tokens=False)
            ids = list(enc["input_ids"])
            if segment_id == 0:
                if prompt_slots is None:
                    prompt_slots = len(ids)
                ids, part_valid = conditioned_prompt_ids(
                    ids, prompt_slots=prompt_slots, pad_id=pad_id,
                    condition=args.prompt_condition,
                )
            else:
                part_valid = [1] * len(ids)
            encoded_parts.append((ids, part_valid, segment_id))
        ids = [bos_id]
        segments = [-1]
        valid = [1]
        for part_ids, part_valid, segment_id in encoded_parts:
            ids.extend(part_ids)
            segments.extend([segment_id] * len(part_ids))
            valid.extend(part_valid)
        ids.append(eos_id)
        segments.append(-1)
        valid.append(1)
        original_count = len(ids)
        ids = ids[:max_length]
        segments = segments[:max_length]
        valid = valid[:max_length]
        sequences.append((ids, segments, valid))
        metadata.append({
            "row_index": row.get("row_index"),
            "document_id": row.get("document_id"),
            "variant": args.variant,
            "prompt_condition": args.prompt_condition,
            "original_token_count": original_count,
            "saved_token_count": len(ids),
            "truncated": original_count > max_length,
            "prompt_token_slots": prompt_slots,
            **sequence_audit(ids, segments, valid),
        })

    prompt_outputs = []
    title_outputs = []
    body_outputs = []
    title_body_outputs = []
    full_outputs = []
    cls_outputs = []
    title_max_outputs = []
    body_max_outputs = []
    title_body_max_outputs = []
    full_max_outputs = []
    prompt_means = []
    truncation = []
    with torch.inference_mode():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start:start + args.batch_size]
            width = max(len(x[0]) for x in batch)
            ids = np.full((len(batch), width), pad_id, dtype=np.int64)
            attention = np.zeros((len(batch), width), dtype=np.int64)
            segments = np.full((len(batch), width), -2, dtype=np.int8)
            valid = np.zeros((len(batch), width), dtype=np.int8)
            for i, (part_ids, part_segments, part_valid) in enumerate(batch):
                n = len(part_ids)
                ids[i, :n] = part_ids
                attention[i, :n] = part_valid
                segments[i, :n] = part_segments
                valid[i, :n] = part_valid
            hidden = model(
                input_ids=torch.from_numpy(ids).to(device),
                attention_mask=torch.from_numpy(attention).to(device),
            ).last_hidden_state
            seg = torch.from_numpy(segments).to(device)
            valid_t = torch.from_numpy(valid).to(device).bool()
            prompt_mask = (seg == 0) & valid_t
            title_mask = (seg == 1) & valid_t
            body_mask = (seg == 2) & valid_t
            title_body_mask = title_mask | body_mask
            full_mask = prompt_mask | title_mask | body_mask
            active_prompt_slots = prompt_slots if args.prompt_condition != "no_prompt_natural" else 0
            prompt_hidden = hidden[:, 1:1 + active_prompt_slots, :].cpu().numpy().astype(np.float32)
            prompt_outputs.append(prompt_hidden)
            prompt_means.append(masked_mean(hidden, prompt_mask).cpu().numpy().astype(np.float32))
            title_outputs.append(masked_mean(hidden, title_mask).cpu().numpy().astype(np.float32))
            body_outputs.append(masked_mean(hidden, body_mask).cpu().numpy().astype(np.float32))
            title_body_outputs.append(masked_mean(hidden, title_body_mask).cpu().numpy().astype(np.float32))
            full_outputs.append(masked_mean(hidden, full_mask).cpu().numpy().astype(np.float32))
            cls_outputs.append(hidden[:, 0, :].cpu().numpy().astype(np.float32))
            title_max_outputs.append(masked_max(hidden, title_mask, torch).cpu().numpy().astype(np.float32))
            body_max_outputs.append(masked_max(hidden, body_mask, torch).cpu().numpy().astype(np.float32))
            title_body_max_outputs.append(masked_max(hidden, title_body_mask, torch).cpu().numpy().astype(np.float32))
            full_max_outputs.append(masked_max(hidden, full_mask, torch).cpu().numpy().astype(np.float32))
            truncation.extend((~((seg == 2) & valid_t).any(dim=1)).cpu().numpy().tolist())
            del hidden

    prompt_tokens = np.concatenate(prompt_outputs, axis=0)
    prompt_mean = np.concatenate(prompt_means, axis=0)
    title_mean = np.concatenate(title_outputs, axis=0)
    body_mean = np.concatenate(body_outputs, axis=0)
    title_body_mean = np.concatenate(title_body_outputs, axis=0)
    full_mean = np.concatenate(full_outputs, axis=0)
    cls = np.concatenate(cls_outputs, axis=0)
    title_max = np.concatenate(title_max_outputs, axis=0)
    body_max = np.concatenate(body_max_outputs, axis=0)
    title_body_max = np.concatenate(title_body_max_outputs, axis=0)
    full_max = np.concatenate(full_max_outputs, axis=0)
    outputs = dict(
        title_mean=title_mean,
        body_mean=body_mean,
        title_body_mean=title_body_mean,
        full_mean=full_mean,
        cls=cls,
        title_max=title_max,
        body_max=body_max,
        title_body_max=title_body_max,
        full_max=full_max,
    )
    if args.prompt_condition == "task_prompt":
        outputs.update(prompt_token_embeddings=prompt_tokens, prompt_mean=prompt_mean)
    save_npz_atomic(args.output_dir / "short_pooling.npz", outputs)
    (args.output_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metadata), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps({
        "model": args.model, "model_path": model_path, "variant": args.variant,
        "prompt_condition": args.prompt_condition,
        "rows": len(rows), "max_length": max_length,
        "prompt_token_slots": prompt_slots,
        "outputs": {
            "title_mean": list(title_mean.shape),
            "body_mean": list(body_mean.shape),
            "title_body_mean": list(title_body_mean.shape),
            "full_mean": list(full_mean.shape),
            "cls": list(cls.shape),
            "title_max": list(title_max.shape),
            "body_max": list(body_max.shape),
            "title_body_max": list(title_body_max.shape),
            "full_max": list(full_max.shape),
            **({
                "prompt_token_embeddings": [len(rows), prompt_slots, int(prompt_tokens.shape[2])],
                "prompt_mean": list(prompt_mean.shape),
            } if args.prompt_condition == "task_prompt" else {}),
        },
        "segment_ids": {"0": "prompt", "1": "title", "2": "body"},
        "note": "Position-matched no-Prompt masks fixed Prompt slots; natural no-Prompt removes them and releases the token budget.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir),
                      "prompt_tokens": list(prompt_tokens.shape),
                      "title_body_mean": list(title_body_mean.shape)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
