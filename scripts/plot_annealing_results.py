"""Plot learning-rate annealing diagnostics and rolling Accuracy results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_reports(paths: list[Path]) -> list[tuple[str, dict]]:
    reports = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reports.append((payload.get("learning_rate_schedule", path.stem), payload))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = load_reports(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {"constant": "#1f77b4", "cosine": "#d62728", "inverse_time": "#2ca02c"}

    for schedule, payload in reports:
        prediction_counts = {}
        prediction_path = payload.get("predictions")
        if prediction_path and Path(prediction_path).is_file():
            prediction_frame = pd.read_parquet(prediction_path, columns=["test_year"])
            prediction_counts = prediction_frame["test_year"].value_counts().to_dict()
        histories = []
        for result in payload.get("results", []):
            history = result.get("early_stopping_history", [])
            if history:
                histories.append(pd.DataFrame(history))
        if histories:
            history = pd.concat(histories, ignore_index=True)
            grouped = history.groupby("epoch", as_index=False).mean(numeric_only=True)
            color = colors.get(schedule)
            axes[0, 0].plot(grouped["epoch"], grouped["validation_accuracy"], label=schedule, color=color)
            axes[0, 1].plot(grouped["epoch"], grouped["train_loss"], label=schedule, color=color)
            axes[1, 0].plot(grouped["epoch"], grouped["learning_rate"], label=schedule, color=color)

        for result in payload.get("results", []):
            summary_rows.append({
                "schedule": schedule,
                "test_year": result.get("test_year"),
                "accuracy": result.get("accuracy"),
                "majority_accuracy": result.get("majority_accuracy"),
                "n_test_rows": result.get("n_test_rows") or prediction_counts.get(result.get("test_year"), 1),
                "validation_accuracy": result.get("validation_accuracy"),
            })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        for schedule, group in summary.groupby("schedule", sort=False):
            group = group.sort_values("test_year")
            axes[1, 1].plot(
                group["test_year"], group["accuracy"] * 100,
                marker="o", label=schedule, color=colors.get(schedule),
            )
            weights = group["n_test_rows"].fillna(1).to_numpy(dtype=float)
            weighted = np.average(group["accuracy"].to_numpy(dtype=float), weights=weights)
            axes[1, 1].axhline(weighted * 100, color=colors.get(schedule), alpha=0.35, linestyle="--")

    axes[0, 0].set_title("Mean validation Accuracy by epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 1].set_title("Mean training loss by epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_title("Learning-rate schedule")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning rate")
    axes[1, 1].set_title("Rolling test Accuracy; dashed = weighted overall")
    axes[1, 1].set_xlabel("Test year")
    axes[1, 1].set_ylabel("Accuracy (%)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Prompt PCA128 + body: learning-rate annealing comparison")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    summary_path = args.output.with_suffix(".csv")
    summary.to_csv(summary_path, index=False)
    aggregate = []
    for schedule, group in summary.groupby("schedule", sort=False):
        weights = group["n_test_rows"].fillna(1).to_numpy(dtype=float)
        accuracy = group["accuracy"].to_numpy(dtype=float)
        aggregate.append({
            "schedule": schedule,
            "weighted_accuracy": float(np.average(accuracy, weights=weights)),
            "mean_annual_accuracy": float(np.mean(accuracy)),
            "annual_accuracy_std": float(np.std(accuracy, ddof=1)) if len(accuracy) > 1 else 0.0,
        })
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"plot": str(args.output), "summary": str(summary_path), "aggregates": aggregate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
