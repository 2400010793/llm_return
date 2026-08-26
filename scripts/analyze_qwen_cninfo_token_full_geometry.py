"""Compare Qwen target-token and article representations on identical CNINFO rows."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROMPTS = ("future_return", "return", "loss", "excess_return", "profit")


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def pair_stats(left_path: Path, right_path: Path, chunk: int) -> dict[str, float]:
    left = np.load(left_path, mmap_mode="r")
    right = np.load(right_path, mmap_mode="r")
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left_path} {right_path}")
    cosines, distances = [], []
    n = sx = sy = sxx = syy = sxy = 0.0
    for start in range(0, len(left), chunk):
        x = np.asarray(left[start:start + chunk], dtype=np.float32)
        y = np.asarray(right[start:start + chunk], dtype=np.float32)
        cosines.append(cosine(x, y))
        distances.append(np.linalg.norm(x - y, axis=1))
        xf, yf = x.ravel().astype(np.float64), y.ravel().astype(np.float64)
        n += len(xf); sx += xf.sum(); sy += yf.sum()
        sxx += np.dot(xf, xf); syy += np.dot(yf, yf); sxy += np.dot(xf, yf)
    c = np.concatenate(cosines); d = np.concatenate(distances)
    cov = sxy - sx * sy / n
    varx = sxx - sx * sx / n; vary = syy - sy * sy / n
    return {
        "rows": len(left), "cosine_mean": float(c.mean()),
        "cosine_median": float(np.median(c)),
        "cosine_p05": float(np.quantile(c, .05)),
        "cosine_p95": float(np.quantile(c, .95)),
        "l2_mean": float(d.mean()),
        "component_pearson": float(cov / np.sqrt(max(varx * vary, 1e-30))),
    }


def mean_daily_spearman(frame: pd.DataFrame, left: str, right: str) -> tuple[float, int]:
    values = []
    for _, group in frame.groupby("entry_date", sort=False):
        if len(group) >= 5 and group[left].nunique() > 1 and group[right].nunique() > 1:
            values.append(group[left].corr(group[right], method="spearman"))
    return (float(np.nanmean(values)), len(values)) if values else (float("nan"), 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-column", default="next_day_open_to_open_return",
        help="Return column used only for the descriptive alignment-factor RankIC.",
    )
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrices = {
        (prompt, kind): args.matrix_root / f"masked_short_{prompt}_{kind}.npy"
        for prompt in PROMPTS for kind in ("article_mean", "target_token")
    }
    missing = [str(path) for path in matrices.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing matrices: " + ", ".join(missing))

    raw = []
    for prompt in PROMPTS:
        raw.append({
            "comparison": "token_vs_article", "left": f"{prompt}_target_token",
            "right": f"{prompt}_article_mean",
            **pair_stats(matrices[prompt, "target_token"], matrices[prompt, "article_mean"], args.chunk_size),
        })
    for kind in ("target_token", "article_mean"):
        for left, right in itertools.combinations(PROMPTS, 2):
            raw.append({
                "comparison": f"cross_prompt_{kind}", "left": left, "right": right,
                **pair_stats(matrices[left, kind], matrices[right, kind], args.chunk_size),
            })
    pd.DataFrame(raw).to_csv(args.output_dir / "raw_embedding_pairwise.csv", index=False)

    panel = pd.read_parquet(args.panel)
    required = [
        "qwen_row_index", "row_index", "stock_id", "entry_date", args.target_column,
    ]
    missing_columns = [column for column in required if column not in panel]
    if missing_columns:
        raise ValueError(f"panel is missing columns: {missing_columns}")
    factors = panel[required].copy().rename(columns={args.target_column: "target_return"})
    for prompt in PROMPTS:
        left = np.load(matrices[prompt, "target_token"], mmap_mode="r")
        right = np.load(matrices[prompt, "article_mean"], mmap_mode="r")
        values = np.empty(len(left), dtype=np.float32)
        for start in range(0, len(left), args.chunk_size):
            values[start:start + args.chunk_size] = cosine(
                np.asarray(left[start:start + args.chunk_size], dtype=np.float32),
                np.asarray(right[start:start + args.chunk_size], dtype=np.float32),
            )
        factors[prompt] = values
    factors.to_parquet(args.output_dir / "announcement_alignment_factors.parquet", index=False)
    stock_day = factors.groupby(["stock_id", "entry_date"], as_index=False).agg(
        target_return=("target_return", "first"),
        n_announcements=("row_index", "size"),
        **{prompt: (prompt, "mean") for prompt in PROMPTS},
    )
    stock_day.to_parquet(args.output_dir / "stock_day_alignment_factors.parquet", index=False)

    pairwise = []
    for left, right in itertools.combinations(PROMPTS, 2):
        daily, days = mean_daily_spearman(stock_day, left, right)
        pairwise.append({
            "left": left, "right": right, "stock_days": len(stock_day),
            "pooled_pearson": pearsonr(stock_day[left], stock_day[right]).statistic,
            "pooled_spearman": spearmanr(stock_day[left], stock_day[right]).statistic,
            "mean_daily_spearman": daily, "daily_spearman_days": days,
        })
    pd.DataFrame(pairwise).to_csv(args.output_dir / "alignment_factor_correlations.csv", index=False)

    predictive = []
    finite = stock_day.dropna(subset=["target_return"])
    for prompt in PROMPTS:
        daily, days = mean_daily_spearman(finite, prompt, "target_return")
        predictive.append({
            "factor": prompt, "stock_days": len(finite), "rank_ic": daily,
            "rank_ic_days": days,
            "pooled_spearman": spearmanr(finite[prompt], finite["target_return"]).statistic,
        })
    pd.DataFrame(predictive).to_csv(args.output_dir / "alignment_factor_predictive.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "format_version": "qwen_cninfo_token_full_geometry_v1",
        "variant": "masked_short", "rows": len(factors),
        "stock_days": len(stock_day), "prompts": list(PROMPTS),
        "target_column": args.target_column,
        "factor": "cosine(target_token_embedding, article_mean_embedding)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(raw).to_string(index=False))
    print(pd.DataFrame(pairwise).to_string(index=False))
    print(pd.DataFrame(predictive).to_string(index=False))


if __name__ == "__main__":
    main()
