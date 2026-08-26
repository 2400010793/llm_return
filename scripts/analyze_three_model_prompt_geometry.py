#!/usr/bin/env python3
"""Measure matched four-prompt token geometry on the frozen common panel."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("roberta", "bge_m3", "qwen3_embedding_8b")
PROMPTS = ("profit", "return", "excess_return", "loss")


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", a, b, optimize=True)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(numerator, denominator, out=np.full(len(a), np.nan), where=denominator > 0)


def centered_kernel_alignment(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    ka = a @ a.T
    kb = b @ b.T
    ka -= ka.mean(axis=0, keepdims=True)
    ka -= ka.mean(axis=1, keepdims=True)
    ka += ka.mean()
    kb -= kb.mean(axis=0, keepdims=True)
    kb -= kb.mean(axis=1, keepdims=True)
    kb += kb.mean()
    denominator = np.linalg.norm(ka) * np.linalg.norm(kb)
    return float(np.sum(ka * kb) / denominator) if denominator else float("nan")


def block_bootstrap(values: pd.DataFrame, column: str, reps: int = 1000) -> tuple[float, float]:
    daily = values.groupby("entry_date", sort=True)[column].mean().dropna().to_numpy(float)
    if len(daily) < 2:
        return float("nan"), float("nan")
    block = min(20, len(daily))
    starts = np.arange(max(1, len(daily) - block + 1))
    rng = np.random.default_rng(42)
    estimates = np.empty(reps, dtype=float)
    blocks = int(np.ceil(len(daily) / block))
    for rep in range(reps):
        sampled = np.concatenate([daily[start:start + block] for start in rng.choice(starts, blocks)])[:len(daily)]
        estimates[rep] = sampled.mean()
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def summarize(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = frame[column].to_numpy(float)
    low, high = block_bootstrap(frame, column)
    return {
        "mean": float(np.nanmean(values)), "median": float(np.nanmedian(values)),
        "std": float(np.nanstd(values)), "q05": float(np.nanquantile(values, 0.05)),
        "q95": float(np.nanquantile(values, 0.95)), "block_ci_low": low,
        "block_ci_high": high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cka-sample", type=int, default=1024)
    args = parser.parse_args()
    panel = pd.read_parquet(args.root / "intersection" / "panel.parquet", columns=["row_index", "entry_date"])
    panel["entry_date"] = pd.to_datetime(panel["entry_date"]).dt.normalize()
    root_manifest = json.loads(
        (args.root / "intersection" / "manifest.json").read_text(encoding="utf-8")
    )
    first_test_year = min(root_manifest["test_years"])
    centering_years = list(range(
        first_test_year - int(root_manifest["history_years"]), first_test_year,
    ))
    baseline = panel["entry_date"].dt.year.isin(centering_years).to_numpy()
    if baseline.sum() == 0:
        raise ValueError(f"no strict first-fold training sample for {centering_years}")
    output = args.root / "geometry"
    output.mkdir(parents=True, exist_ok=True)
    detail_rows, summary_rows, cka_rows, token_body_rows = [], [], [], []

    for model in MODELS:
        matrices = {
            prompt: np.asarray(np.load(args.root / "matrices" / model / prompt / "token" / "matrix.npy", mmap_mode="r"), dtype=np.float32)
            for prompt in PROMPTS
        }
        means = {prompt: matrix[baseline].mean(axis=0) for prompt, matrix in matrices.items()}
        scales = {
            prompt: np.maximum(matrix[baseline].std(axis=0), 1e-6)
            for prompt, matrix in matrices.items()
        }
        rng = np.random.default_rng(42)
        sample = np.sort(rng.choice(len(panel), min(args.cka_sample, len(panel)), replace=False))
        for left, right in combinations(PROMPTS, 2):
            a, b = matrices[left], matrices[right]
            raw = row_cosine(a, b)
            centered = row_cosine(a - means[left], b - means[right])
            standardized_l2 = np.linalg.norm(
                (a - means[left]) / scales[left] - (b - means[right]) / scales[right], axis=1,
            ) / np.sqrt(a.shape[1])
            detail = panel.copy()
            detail["model"], detail["left_prompt"], detail["right_prompt"] = model, left, right
            detail["raw_cosine"], detail["centered_cosine"] = raw, centered
            detail["standardized_l2"] = standardized_l2
            detail_rows.append(detail)
            row = {"model": model, "left_prompt": left, "right_prompt": right, "rows": len(detail)}
            for metric in ("raw_cosine", "centered_cosine", "standardized_l2"):
                row.update({f"{metric}_{key}": value for key, value in summarize(detail, metric).items()})
            summary_rows.append(row)
            cka_rows.append({
                "model": model, "left_prompt": left, "right_prompt": right,
                "sample_rows": len(sample), "linear_cka": centered_kernel_alignment(a[sample], b[sample]),
            })
        for prompt in PROMPTS:
            token = matrices[prompt]
            body = np.asarray(np.load(args.root / "matrices" / model / prompt / "body" / "matrix.npy", mmap_mode="r"), dtype=np.float32)
            values = row_cosine(token, body)
            frame = panel.copy()
            frame["token_body_cosine"] = values
            token_body_rows.append({"model": model, "prompt": prompt, **summarize(frame, "token_body_cosine")})

    detail = pd.concat(detail_rows, ignore_index=True)
    detail.to_parquet(output / "pairwise_token_geometry.parquet", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "pairwise_token_geometry_summary.csv", index=False)
    pd.DataFrame(cka_rows).to_csv(output / "linear_cka.csv", index=False)
    pd.DataFrame(token_body_rows).to_csv(output / "token_body_cosine.csv", index=False)
    audit = {
        "format_version": "three_model_prompt_geometry_pca_v2", "rows": len(panel),
        "dataset": root_manifest["dataset"], "centering_years": centering_years,
        "cka_sample": min(args.cka_sample, len(panel)),
        "block_bootstrap_reps": 1000, "block_length_dates": 20,
    }
    (output / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output / "COMPLETED").write_text("three_model_prompt_geometry_pca_v2\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
