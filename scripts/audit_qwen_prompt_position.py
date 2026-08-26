"""Audit Qwen prompt-token embeddings by their recorded absolute prompt position."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--fixed-position", type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    base = args.root / args.dataset / args.prompt / args.variant
    rows = []
    for shard in sorted(base.glob("shard-*")):
        meta_path = shard / "metadata.jsonl"
        emb_path = shard / "prompt_token_embeddings.npy"
        if not meta_path.is_file() or not emb_path.is_file():
            continue
        emb = np.load(emb_path, mmap_mode="r").astype(np.float32)
        with meta_path.open(encoding="utf-8") as fh:
            meta = [json.loads(line) for line in fh]
        if len(meta) != emb.shape[0]:
            raise ValueError(f"row mismatch: {shard}: {len(meta)} vs {emb.shape[0]}")
        for i, item in enumerate(meta):
            pos = int(item["prompt_start_zero_based"])
            group = "fixed" if pos == args.fixed_position else "variable"
            x = emb[i]
            rows.append((group, pos, x))

    if not rows:
        raise ValueError(f"no completed shards under {base}")
    if args.fixed_position is None:
        # The largest observed start is the natural full-context/truncation
        # position for this prompt. It is prompt-tokenization dependent.
        args.fixed_position = max(pos for _, pos, _ in rows)
    n_tokens = rows[0][2].shape[0]
    groups = {"fixed": [], "variable": []}
    positions: dict[int, int] = {}
    for group, pos, x in rows:
        groups[group].append(x)
        positions[pos] = positions.get(pos, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "dataset", "prompt", "variant", "group", "position", "rows",
            "mean_token_norm", "mean_token_std", "mean_abs_value",
        ])
        writer.writeheader()
        for group, arrays in groups.items():
            if not arrays:
                continue
            x = np.concatenate(arrays, axis=0)
            writer.writerow({
                "dataset": args.dataset, "prompt": args.prompt,
                "variant": args.variant, "group": group,
                "position": args.fixed_position if group == "fixed" else "other",
                "rows": len(x), "mean_token_norm": float(np.linalg.norm(x, axis=1).mean()),
                "mean_token_std": float(x.std(axis=0).mean()),
                "mean_abs_value": float(np.abs(x).mean()),
            })
        writer.writerow({
            "dataset": args.dataset, "prompt": args.prompt,
            "variant": args.variant, "group": "position_counts",
            "position": json.dumps(dict(sorted(positions.items())), ensure_ascii=False),
            "rows": sum(positions.values()), "mean_token_norm": "", "mean_token_std": "",
            "mean_abs_value": "",
        })

    # A compact JSON sidecar makes the audit reproducible without retaining arrays.
    fixed_mean = np.concatenate(groups["fixed"], axis=0).mean(axis=0) if groups["fixed"] else None
    variable_mean = np.concatenate(groups["variable"], axis=0).mean(axis=0) if groups["variable"] else None
    centroid_cosine = None
    if fixed_mean is not None and variable_mean is not None:
        denom = float(np.linalg.norm(fixed_mean) * np.linalg.norm(variable_mean))
        centroid_cosine = float(np.dot(fixed_mean, variable_mean) / denom) if denom else None
    summary = {
        "dataset": args.dataset, "prompt": args.prompt, "variant": args.variant,
        "fixed_position": args.fixed_position, "rows": len(rows), "token_count": n_tokens,
        "fixed_rows": len(groups["fixed"]), "variable_rows": len(groups["variable"]),
        "fixed_variable_centroid_cosine": centroid_cosine,
        "position_counts": dict(sorted(positions.items())),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
