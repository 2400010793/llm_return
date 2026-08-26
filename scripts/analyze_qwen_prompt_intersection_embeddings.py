"""Compare the currently available Qwen prompt embeddings on their exact intersection.

This deliberately works on completed shard outputs, so it can be run before
all 1000 shards finish. Each variant is analyzed independently. Token indices
are resolved from each prompt's own prompt_tokens.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROMPTS = ("future_return", "return", "profit", "excess_return", "loss")
TARGETS = {"future_return": ("未来", "收益"), "return": ("收益",), "profit": ("盈利",), "excess_return": ("超额", "收益"), "loss": ("亏损",)}


def completed_shards(root: Path, prompt: str, variant: str) -> list[Path]:
    return sorted((root / prompt / variant).glob("shard-*/COMPLETED"), key=lambda p: int(p.parent.name.split("-")[1]))


def index_prompt(root: Path, prompt: str, variant: str) -> tuple[dict[int, tuple[Path, int]], list[int]]:
    index: dict[int, tuple[Path, int]] = {}
    for marker in completed_shards(root, prompt, variant):
        shard = marker.parent
        metadata = [json.loads(line) for line in (shard / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for position, row in enumerate(metadata):
            row_index = int(row["row_index"])
            if row_index in index:
                raise ValueError(f"duplicate row_index={row_index}: {prompt}/{variant}")
            index[row_index] = (shard, position)
    return index, sorted(index)


def representation(shard: Path, positions: list[int], kind: str, prompt: str) -> np.ndarray:
    if kind == "article_mean":
        return np.asarray(np.load(shard / "article_mean_embeddings.npy", mmap_mode="r")[positions], dtype=np.float32)
    token_info = json.loads((shard / "prompt_tokens.json").read_text(encoding="utf-8"))
    indices = [i for i, token in enumerate(token_info["tokens"]) if token in TARGETS[prompt]]
    if not indices:
        raise ValueError(f"target token not found for {prompt}: {token_info['tokens']}")
    tokens = np.asarray(np.load(shard / "prompt_token_embeddings.npy", mmap_mode="r")[positions], dtype=np.float32)
    return tokens[:, indices, :].mean(axis=1)


def load_rep(root: Path, prompt: str, variant: str, common: list[int], locations: dict[int, tuple[Path, int]], kind: str) -> np.ndarray:
    by_shard: dict[Path, list[tuple[int, int]]] = {}
    for output_position, row_index in enumerate(common):
        shard, position = locations[row_index]
        by_shard.setdefault(shard, []).append((output_position, position))
    first = next(iter(by_shard.values()))
    hidden = int(np.load(next(iter(locations.values()))[0] / "article_mean_embeddings.npy", mmap_mode="r").shape[-1])
    result = np.empty((len(common), hidden), dtype=np.float32)
    for shard, pairs in by_shard.items():
        output_positions, positions = zip(*pairs)
        result[list(output_positions)] = representation(shard, list(positions), kind, prompt)
    return result


def compare(x: np.ndarray, y: np.ndarray, left: str, right: str) -> dict[str, object]:
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    cosine = np.sum(x * y, axis=1) / np.maximum(nx * ny, 1e-12)
    distance = np.linalg.norm(x - y, axis=1)
    return {"left": left, "right": right, "rows": len(x), "cosine_mean": cosine.mean(), "cosine_median": np.median(cosine), "cosine_p05": np.quantile(cosine, .05), "cosine_p95": np.quantile(cosine, .95), "l2_mean": distance.mean(), "component_pearson": pearsonr(x.ravel(), y.ravel())[0], "cosine_l2_spearman": spearmanr(cosine, distance).statistic}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--variants", default="short,masked_short")
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    for variant in [x.strip() for x in args.variants.split(",") if x.strip()]:
        indexed = {prompt: index_prompt(args.root, prompt, variant)[0] for prompt in PROMPTS}
        common = sorted(set.intersection(*(set(x) for x in indexed.values())))
        raw_rows = []
        matrices_by_kind = {}
        for kind in ("article_mean", "target_token"):
            reps = {prompt: load_rep(args.root, prompt, variant, common, indexed[prompt], kind) for prompt in PROMPTS}
            matrices_by_kind[kind] = reps
            for i, left in enumerate(PROMPTS):
                for right in PROMPTS[i + 1:]:
                    raw_rows.append(compare(reps[left], reps[right], f"{left}__{kind}", f"{right}__{kind}"))
            for prompt in PROMPTS:
                raw_rows.append(compare(reps[prompt], load_rep(args.root, prompt, variant, common, indexed[prompt], "article_mean" if kind == "target_token" else "target_token"), f"{prompt}__{kind}", f"{prompt}__other"))
        out = args.output_root / variant
        out.mkdir(parents=True, exist_ok=True)
        # Persist the current intersection as reusable regression inputs.  The
        # matrices are not complete Qwen datasets; the audit records the exact
        # completed-shard intersection used to create them.
        metadata = pd.DataFrame({"row_index": common})
        metadata.to_parquet(out / "intersection_metadata.parquet", index=False)
        for prompt in PROMPTS:
            np.save(out / f"{prompt}_article_mean.npy", matrices_by_kind["article_mean"][prompt].astype(np.float16))
            np.save(out / f"{prompt}_target_token.npy", matrices_by_kind["target_token"][prompt].astype(np.float16))
        pd.DataFrame(raw_rows).to_csv(out / "raw_embedding_pairwise.csv", index=False)
        pd.DataFrame([{"variant": variant, "common_rows": len(common), "completed_shards": ";".join(f"{p}:{len(completed_shards(args.root,p,variant))}" for p in PROMPTS)}]).to_csv(out / "intersection_audit.csv", index=False)
        print(variant, "common_rows", len(common))
        print(pd.DataFrame(raw_rows).to_string(index=False))


if __name__ == "__main__":
    main()