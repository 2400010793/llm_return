"""Position-controlled classification on completed Qwen Sina shards only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def load(root: Path, prompt: str, variant: str, panel: pd.DataFrame):
    base = root / "sina" / prompt / variant
    xs, rows, positions = [], [], []
    for shard in sorted(base.glob("shard-*/COMPLETED")):
        d = shard.parent
        emb = np.load(d / "prompt_token_embeddings.npy", mmap_mode="r")
        meta = [json.loads(line) for line in (d / "metadata.jsonl").open(encoding="utf-8")]
        if len(meta) != emb.shape[0]:
            raise ValueError(f"row mismatch in {d}")
        xs.append(np.asarray(emb.mean(axis=1), dtype=np.float32))
        rows.extend(int(m["row_index"]) for m in meta)
        positions.extend(int(m["prompt_start_zero_based"]) for m in meta)
    x = np.concatenate(xs, axis=0)
    frame = pd.DataFrame({"row_index": rows, "prompt_start": positions})
    frame = frame.merge(panel[["row_index", "entry_date", "next_day_label"]], on="row_index", how="inner")
    if len(frame) != len(x):
        keep = pd.Series(rows).isin(set(frame.row_index)).to_numpy()
        x = x[keep]
    # Merge ordering is preserved by pandas for unique row_index in this panel.
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    return x, frame.reset_index(drop=True)


def fit_eval(x, f, fixed_position=None):
    if fixed_position is not None:
        keep = f.prompt_start.eq(fixed_position).to_numpy()
        x, f = x[keep], f.loc[keep].reset_index(drop=True)
    year = f.entry_date.dt.year.to_numpy()
    y = pd.to_numeric(f.next_day_label, errors="coerce").to_numpy()
    valid = np.isfinite(y)
    x, f, year, y = x[valid], f.loc[valid], year[valid], y[valid].astype(int)
    train, val, test = year <= 2023, np.isin(year, [2024, 2025]), year == 2026
    out = {"sample": len(y), "fixed_position": fixed_position}
    if train.sum() < 100 or val.sum() < 50 or test.sum() < 20:
        return out
    scaler = StandardScaler().fit(x[train])
    pca = PCA(n_components=min(32, x.shape[1], train.sum() - 1), svd_solver="randomized", random_state=42).fit(scaler.transform(x[train]))
    ztr, zv, zte = pca.transform(scaler.transform(x[train])), pca.transform(scaler.transform(x[val])), pca.transform(scaler.transform(x[test]))
    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=300, random_state=42).fit(ztr, y[train])
        score = balanced_accuracy_score(y[val], m.predict(zv))
        if best is None or score > best[0]: best = (score, c)
    # Refit selected linear model on train+validation after selection.
    fit = train | val
    scaler = StandardScaler().fit(x[fit])
    pca = PCA(n_components=min(32, x.shape[1], fit.sum() - 1), svd_solver="randomized", random_state=42).fit(scaler.transform(x[fit]))
    m = LogisticRegression(C=best[1], class_weight="balanced", max_iter=300, random_state=42).fit(pca.transform(scaler.transform(x[fit])), y[fit])
    pred = m.predict(pca.transform(scaler.transform(x[test])))
    prob = m.predict_proba(pca.transform(scaler.transform(x[test])))[:, 1]
    out.update({"train": int(train.sum()), "validation": int(val.sum()), "test": int(test.sum()), "best_C": best[1], "val_balanced_accuracy": best[0], "test_accuracy": accuracy_score(y[test], pred), "test_balanced_accuracy": balanced_accuracy_score(y[test], pred), "test_auc": roc_auc_score(y[test], prob) if len(np.unique(y[test])) == 2 else np.nan, "positive_rate": float(y[test].mean()), "predicted_positive_rate": float(pred.mean())})
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--panel", type=Path, required=True); p.add_argument("--prompt", default="return"); p.add_argument("--variant", default="short"); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); panel = pd.read_parquet(a.panel)
    x, f = load(a.root, a.prompt, a.variant, panel)
    fixed = int(f.prompt_start.value_counts().index[0])
    results = [fit_eval(x, f, None), fit_eval(x, f, fixed)]
    for r, name in zip(results, ("all_positions", "fixed_position")): r["sample_group"] = name; r["prompt"] = a.prompt; r["variant"] = a.variant
    a.output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(results).to_csv(a.output, index=False)


if __name__ == "__main__": main()
