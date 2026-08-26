"""Encode all structured text variants in one model process.

Outputs six embedding variants plus raw financial keyword features. The model is
loaded once; batches are encoded one variant at a time and written to memmaps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_cninfo_prompt_inputs import (  # noqa: E402
    LONG_TEMPLATE,
    MASKED_LONG_TEMPLATE,
    MASKED_SHORT_TEMPLATE,
    SHORT_TEMPLATE,
    mask_identity_and_time,
)
from src.text.lexicon_features import lexicon_score  # noqa: E402

VARIANTS = (
    "short",
    "masked_short",
    "title",
    "body",
    "title_body",
)


def records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def make_texts(row: dict) -> dict[str, str]:
    body = str(row.get("body", row.get("text_model", row.get("text", ""))) or "")
    title = str(row.get("title", row.get("title_clean_final", "")) or "")
    name = str(row.get("stock_name", "") or row.get("stock_id", ""))
    stock_id = str(row.get("stock_id", "") or "")
    masked_body = mask_identity_and_time(body, name, stock_id)
    masked_title = mask_identity_and_time(title, name, stock_id)
    return {
        "short": SHORT_TEMPLATE.format(stock_name=name, text=body),
        "masked_short": MASKED_SHORT_TEMPLATE.format(text=masked_body),
        "title": title,
        "body": body,
        "title_body": title + "\n" + body,
        "masked_title_body": MASKED_LONG_TEMPLATE.format(
            title=masked_title, text=masked_body
        ),
    }


def encode_ollama(texts: list[str], args: argparse.Namespace) -> np.ndarray:
    from src.text.ollama_client import ollama_embed

    return np.asarray(
        ollama_embed(
            texts,
            model=args.model,
            base_url=args.base_url,
            timeout=args.timeout,
        ),
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("ollama", "transformer"), required=True)
    parser.add_argument("--model", default="qwen3-embedding:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--transformer-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The source order is the canonical row order for every saved matrix.
    total = sum(1 for _ in records(args.input))
    if total == 0:
        raise ValueError("input contains no records")

    model = None
    tokenizer = None
    torch = None
    if args.backend == "transformer":
        import torch as torch_module
        from transformers import AutoModel, AutoTokenizer

        torch = torch_module
        target = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(target).eval()

    handles: dict[str, np.memmap] = {}
    metadata: list[dict] = []
    keyword_rows: list[dict] = []
    row_batch: list[dict] = []

    def encode(texts: list[str]) -> np.ndarray:
        if args.backend == "ollama":
            return encode_ollama(texts, args)
        assert model is not None and tokenizer is not None and torch is not None
        target = next(model.parameters()).device
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(target) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.cpu().numpy().astype(np.float32)

    def flush(batch: list[dict]) -> None:
        if not batch:
            return
        texts = {variant: [make_texts(row)[variant] for row in batch] for variant in VARIANTS}
        matrices = {variant: encode(values) for variant, values in texts.items()}
        start = len(metadata)
        end = start + len(batch)
        for variant, matrix in matrices.items():
            if variant not in handles:
                path = args.output_dir / f"{variant}.npy"
                handles[variant] = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=(total, matrix.shape[1])
                )
            handles[variant][start:end] = matrix
        for row in batch:
            idx = len(metadata)
            metadata.append({
                "row_index": idx,
                "document_id": row.get("document_id"),
                "stock_id": str(row.get("stock_id", "")),
                "announcement_date": row.get("announcement_date"),
                "text_sha256": row.get("text_sha256", ""),
            })
            keyword_rows.append(lexicon_score(str(row.get("body", row.get("text_model", "")) or "")))

    for row in records(args.input):
        row_batch.append(row)
        if len(row_batch) >= args.batch_size:
            flush(row_batch)
            row_batch = []
    flush(row_batch)
    for handle in handles.values():
        handle.flush()

    metadata_frame = pd.DataFrame(metadata)
    keyword_frame = pd.DataFrame(keyword_rows)
    metadata_frame.to_parquet(args.output_dir / "metadata.parquet", index=False)
    keyword_frame.to_parquet(args.output_dir / "financial_keywords.parquet", index=False)
    # Raw concatenation is saved for convenience; scaling must be fitted on train only.
    body = np.load(args.output_dir / "body.npy", mmap_mode="r")
    combined = np.lib.format.open_memmap(
        args.output_dir / "body_plus_financial_keywords.npy",
        mode="w+", dtype=np.float32,
        shape=(total, body.shape[1] + keyword_frame.shape[1]),
    )
    combined[:, :body.shape[1]] = body
    combined[:, body.shape[1]:] = keyword_frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    combined.flush()
    (args.output_dir / "summary.json").write_text(
        json.dumps({
            "rows": total,
            "backend": args.backend,
            "model": args.model,
            "variants": list(VARIANTS) + ["body_plus_financial_keywords"],
            "dimensions": {name: int(np.load(args.output_dir / f"{name}.npy", mmap_mode="r").shape[1]) for name in VARIANTS},
            "keyword_columns": list(keyword_frame.columns),
            "note": "Keyword scaling must be fitted on the training split only.",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rows": total, "output_dir": str(args.output_dir), "variants": list(VARIANTS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
