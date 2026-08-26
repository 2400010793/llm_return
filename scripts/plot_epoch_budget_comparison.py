"""Plot validation epoch selection against final-fit epoch budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = payload["results"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for result in results:
        history = result["history"]
        axes[0].plot(
            [row["epoch"] for row in history],
            [row["validation_accuracy"] * 100 for row in history],
            label=(f'{result["multiplier"]:.2f}x; final={result["final_fit_epoch"]}'),
        )
    axes[0].set_title("Validation Accuracy and selected epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Accuracy (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels = [f'{result["multiplier"]:.2f}x' for result in results]
    values = [result["test_accuracy"] * 100 for result in results]
    axes[1].bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].axhline(results[0]["test_majority_accuracy"] * 100, color="#777", linestyle="--", label="majority baseline")
    axes[1].set_title(f'Test Accuracy ({payload["protocol"]["test_year"]})')
    axes[1].set_xlabel("Final-fit epoch multiplier")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.suptitle("Validation-selected epoch vs extended final-fit budget")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(str(args.output))


if __name__ == "__main__":
    main()
