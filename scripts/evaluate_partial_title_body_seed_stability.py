"""Evaluate completed title/body seed models on one shared test year."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import (
    align_embeddings_to_panel,
    load_pooled_embeddings,
)
from src.evaluation.classification import evaluate_binary_classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--test-year", type=int, default=2018)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundles = sorted(args.artifact_root.glob("*.artifacts"))
    bundles.extend(sorted((args.artifact_root / "seed_stability").glob("*.artifacts")))
    completed = [
        bundle for bundle in bundles
        if (bundle / f"test_year_{args.test_year}" / "final_model.joblib").is_file()
    ]
    if not completed:
        raise ValueError(f"no completed test-year models under {args.artifact_root}")

    panel = pd.read_parquet(args.panel)
    embeddings = load_pooled_embeddings(
        args.embedding_root,
        model="roberta", variant="short", feature="title_body_mean",
        require_complete_rows=903665, max_matrix_gib=8.0,
    )
    frame, matrix = align_embeddings_to_panel(panel, embeddings)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    valid = frame["entry_date"].notna().to_numpy()
    frame, matrix = frame.loc[valid].copy(), matrix[valid]
    order = np.argsort(frame["entry_date"].to_numpy(), kind="stable")
    frame, matrix = frame.iloc[order].reset_index(drop=True), matrix[order]
    test_mask = frame["entry_date"].dt.year.eq(args.test_year).to_numpy()
    x_test = matrix[test_mask]
    target = pd.to_numeric(
        frame.loc[test_mask, "next_day_return"], errors="coerce"
    ).to_numpy(dtype=float)

    rows: list[dict] = []
    labels_by_group: dict[str, list[np.ndarray]] = {}
    for bundle in completed:
        spec = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
        experiment = spec["experiment"]
        metric = str(experiment["early_stopping_metric"])
        if metric not in {"accuracy", "balanced_accuracy"}:
            continue
        seed = int(experiment["seed"])
        fold = bundle / f"test_year_{args.test_year}"
        preprocessor = joblib.load(fold / "all_train_preprocessor.joblib")
        if any(value is not None for value in preprocessor.values()):
            raise ValueError(f"unexpected preprocessing for title_body_mean: {bundle}")
        model = joblib.load(fold / "final_model.joblib")
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = evaluate_binary_classification(target, probabilities)
        predicted = probabilities >= 0.5
        labels_by_group.setdefault(metric, []).append(predicted)
        history = json.loads(
            (fold / "mlp_training_history.json").read_text(encoding="utf-8")
        )
        rows.append({
            "selection_metric": metric,
            "seed": seed,
            "test_year": args.test_year,
            "selected_epoch": int(history["selected_params"]["selected_epoch"]),
            **{key: float(value) for key, value in metrics.items()},
            "artifact_bundle": str(bundle),
        })

    summaries = []
    for metric, group in pd.DataFrame(rows).groupby("selection_metric", sort=True):
        accuracy = group["accuracy"]
        agreements = [
            float(np.mean(left == right))
            for left, right in itertools.combinations(labels_by_group[metric], 2)
        ]
        summaries.append({
            "selection_metric": metric,
            "seed_count": int(len(group)),
            "accuracy_mean": float(accuracy.mean()),
            "accuracy_std": float(accuracy.std(ddof=1)),
            "accuracy_min": float(accuracy.min()),
            "accuracy_max": float(accuracy.max()),
            "accuracy_range_pp": float(100.0 * (accuracy.max() - accuracy.min())),
            "pairwise_label_agreement_mean": (
                float(np.mean(agreements)) if agreements else None
            ),
            "pairwise_label_agreement_min": (
                float(np.min(agreements)) if agreements else None
            ),
        })
    payload = {
        "test_year": args.test_year,
        "models_evaluated": len(rows),
        "summaries": summaries,
        "per_seed": sorted(rows, key=lambda row: (row["selection_metric"], row["seed"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
