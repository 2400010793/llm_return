"""Plot interim title/body MLP training and seed-stability diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("accuracy", "balanced_accuracy")
COLORS = {42: "#0072B2", 43: "#E69F00", 44: "#009E73", 45: "#D55E00", 46: "#CC79A7"}


def bundles(root: Path) -> list[Path]:
    values = sorted(root.glob("*.artifacts"))
    values.extend(sorted((root / "seed_stability").glob("*.artifacts")))
    return values


def identity(bundle: Path) -> tuple[str, int]:
    spec = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
    experiment = spec["experiment"]
    return str(experiment["early_stopping_metric"]), int(experiment["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_root.mkdir(parents=True, exist_ok=True)
    histories: dict[tuple[str, int], pd.DataFrame] = {}
    for bundle in bundles(args.root):
        metric, seed = identity(bundle)
        if metric not in METRICS:
            continue
        path = bundle / "test_year_2018" / "mlp_training_history.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            histories[(metric, seed)] = pd.DataFrame(payload["selected_epoch_history"])

    fields = (
        ("validation_accuracy", "Validation Accuracy"),
        ("validation_balanced_accuracy", "Validation Balanced Accuracy"),
        ("validation_auc", "Validation AUC"),
        ("train_loss", "Training loss"),
    )
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), constrained_layout=True)
    for column, metric in enumerate(METRICS):
        for row, (field, label) in enumerate(fields):
            axis = axes[row, column]
            for seed in range(42, 47):
                history = histories.get((metric, seed))
                if history is None or field not in history:
                    continue
                axis.plot(
                    history["epoch"], history[field],
                    color=COLORS[seed], linewidth=1.5, label=f"seed {seed}",
                )
                selected = history.loc[history["selection_score"].idxmax()]
                axis.scatter(
                    [selected["epoch"]], [selected[field]],
                    color=COLORS[seed], s=24, zorder=3,
                )
            axis.set_title(f"2018 / select by {metric}: {label}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            if row == 0:
                axis.legend(ncol=3, fontsize=8)
    training_path = args.output_root / "title_body_2018_training_curves.png"
    fig.savefig(training_path, dpi=180)
    plt.close(fig)

    partial_rows = []
    for year in (2018, 2019):
        path = args.root / f"partial_{year}_seed_stability.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        partial_rows.extend(payload["per_seed"])
    frame = pd.DataFrame(partial_rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    markers = {"accuracy": "o", "balanced_accuracy": "s"}
    for metric in METRICS:
        group = frame.loc[frame["selection_metric"].eq(metric)]
        for year, year_group in group.groupby("test_year"):
            axes[0, 0].plot(
                year_group["seed"], year_group["accuracy"] * 100,
                marker=markers[metric], label=f"{metric} / {year}",
            )
            axes[0, 1].plot(
                year_group["seed"], year_group["predicted_positive_rate"] * 100,
                marker=markers[metric], label=f"{metric} / {year}",
            )
            axes[1, 0].plot(
                year_group["seed"], year_group["selected_epoch"],
                marker=markers[metric], label=f"{metric} / {year}",
            )
        summary = group.groupby("test_year", as_index=False).agg(
            mean=("accuracy", "mean"), std=("accuracy", "std"),
            minimum=("accuracy", "min"), maximum=("accuracy", "max"),
        )
        axes[1, 1].errorbar(
            summary["test_year"], summary["mean"] * 100,
            yerr=summary["std"] * 100, marker=markers[metric], capsize=4,
            label=metric,
        )
    baselines = frame.groupby("test_year")["majority_accuracy"].first()
    for year, baseline in baselines.items():
        axes[1, 1].scatter([year], [baseline * 100], color="black", marker="x", s=55)
    axes[0, 0].set_title("Test Accuracy by seed")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Predicted positive rate by seed")
    axes[0, 1].set_ylabel("Predicted positive (%)")
    axes[1, 0].set_title("Validation-selected epoch by seed")
    axes[1, 0].set_ylabel("Epoch")
    axes[1, 1].set_title("Test Accuracy mean +/- one seed SD; x = majority")
    axes[1, 1].set_ylabel("Accuracy (%)")
    for axis in axes.flat:
        axis.set_xlabel("Seed" if axis is not axes[1, 1] else "Test year")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    stability_path = args.output_root / "title_body_2018_2019_seed_stability.png"
    fig.savefig(stability_path, dpi=180)
    plt.close(fig)
    frame.to_csv(args.output_root / "title_body_partial_seed_metrics.csv", index=False)
    print(json.dumps({
        "training_curves": str(training_path),
        "seed_stability": str(stability_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
