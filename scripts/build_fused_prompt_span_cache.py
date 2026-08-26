"""Build an aligned concatenation of two existing semantic span caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--semantic-name", default="stock_span")
    p.add_argument("--expected-rows", type=int, default=None)
    args = p.parse_args()
    left = json.loads((args.left / "manifest.json").read_text())
    right = json.loads((args.right / "manifest.json").read_text())
    if left["rows"] != right["rows"]:
        raise ValueError("span caches have different row counts")
    if left["phrase"] != right["phrase"] or (args.expected_rows is not None and left["rows"] != args.expected_rows):
        raise ValueError("unexpected span cache alignment")
    args.output.mkdir(parents=True, exist_ok=True)
    for variant in ("short", "masked_short"):
        a = np.load(args.left / f"{variant}_{args.semantic_name}.npy", mmap_mode="r")
        b = np.load(args.right / f"{variant}_{args.semantic_name}.npy", mmap_mode="r")
        if a.shape[0] != b.shape[0]:
            raise ValueError(f"{variant} row counts differ: {a.shape} vs {b.shape}")
        out = np.lib.format.open_memmap(
            args.output / f"{variant}_{args.semantic_name}.npy",
            mode="w+", dtype=np.float32, shape=(a.shape[0], a.shape[1] + b.shape[1]),
        )
        out[:, :a.shape[1]] = a
        out[:, a.shape[1]:] = b
        out.flush()
    manifest = dict(left)
    manifest.update({
        "format_version": "prompt_semantic_span_cache_fusion_v1",
        "model": "roberta_bge_m3_fusion",
        "hidden_size": int(left["hidden_size"] + right["hidden_size"]),
        "source_models": [left["model"], right["model"]],
    })
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (args.output / "COMPLETED").write_text("prompt_span_fusion_v1\n")
    print(json.dumps({"output": str(args.output), "rows": left["rows"], "hidden_size": manifest["hidden_size"]}))


if __name__ == "__main__":
    main()
