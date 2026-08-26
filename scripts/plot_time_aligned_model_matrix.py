"""Plot the unified Accuracy matrix as a publication-friendly PNG."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


CLASSIFIER_ORDER = [
    "logistic", "linear_svm", "nb_svm", "multinomial_nb", "complement_nb",
    "sgd", "random_forest", "mlp", "knn",
]


def pretty_classifier(value: str) -> str:
    return {
        "logistic": "Logistic",
        "linear_svm": "Linear SVM",
        "nb_svm": "NB-SVM",
        "multinomial_nb": "Multinomial NB",
        "complement_nb": "Complement NB",
        "sgd": "SGD",
        "random_forest": "Random Forest",
        "mlp": "MLP",
        "knn": "KNN",
    }.get(value, value)


def pretty_column(model: str, prompt: str, reducer: str) -> str:
    model_name = {
        "char_tfidf": "Char TF-IDF",
        "word_tfidf": "Word TF-IDF",
        "word_count": "Word Count",
        "chinese_roberta": "RoBERTa",
        "bge_m3": "BGE-M3",
        "qwen": "Qwen",
    }.get(model, model)
    prompt_name = {
        "none": "",
        "existing": "existing",
        "plain": "plain",
        "short": "short",
        "long": "long",
        "masked_short": "masked-short",
        "masked_long": "masked-long",
    }.get(prompt, prompt)
    reducer_name = "none" if reducer == "none" else reducer.upper()
    if prompt_name:
        return f"{model_name}\n{prompt_name}\n{reducer_name}"
    return f"{model_name}\n{reducer_name}"


def read_matrix(csv_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    classifiers = [name for name in CLASSIFIER_ORDER if any(row["classifier"] == name for row in rows)]
    columns: list[tuple[str, str, str]] = []
    for row in rows:
        column = (row["model"], row["prompt_state"], row["reducer"])
        if column not in columns:
            columns.append(column)
    columns.sort(key=lambda item: (item[0], item[1], item[2]))
    # Put sparse representations first, then existing embeddings, then prompts.
    family_order = {"char_tfidf": 0, "word_tfidf": 1, "word_count": 2,
                    "chinese_roberta": 3, "bge_m3": 4, "qwen": 5}
    prompt_order = {"none": 0, "existing": 0, "plain": 1, "short": 2,
                    "long": 3, "masked_short": 4, "masked_long": 5}
    columns.sort(key=lambda item: (family_order.get(item[0], 99), prompt_order.get(item[1], 99), item[2]))
    lookup = {(row["classifier"], row["model"], row["prompt_state"], row["reducer"]): row for row in rows}
    matrix = np.full((len(classifiers), len(columns)), np.nan)
    for i, classifier in enumerate(classifiers):
        for j, column in enumerate(columns):
            value = lookup.get((classifier, *column), {}).get("accuracy", "")
            if value not in ("", "—"):
                matrix[i, j] = float(value)
    labels = [pretty_column(*column) for column in columns]
    return [pretty_classifier(name) for name in classifiers], labels, matrix


def plot(csv_path: Path, output_path: Path) -> None:
    classifiers, columns, matrix = read_matrix(csv_path)
    n_rows, n_columns = matrix.shape
    cmap = LinearSegmentedColormap.from_list("accuracy", ["#b2182b", "#f7f7f7", "#2166ac"])
    cmap.set_bad("#d9d9d9")
    fig_width = max(24, n_columns * 1.25)
    fig, ax = plt.subplots(figsize=(fig_width, 9.5), constrained_layout=True)
    norm = Normalize(vmin=0.45, vmax=0.55)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    ax.set_xticks(np.arange(n_columns))
    ax.set_xticklabels(columns, fontsize=8, linespacing=1.25)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(classifiers, fontsize=11)
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True, pad=8)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("Model / Prompt state / Reducer", fontsize=12, labelpad=18)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Classifier", fontsize=12, labelpad=12)
    ax.set_title(
        "Time-aligned classification matrix — Test Accuracy\n"
        "Train: event_return_3d → Evaluate: next_day_return | Gray: unavailable",
        fontsize=15, pad=70,
    )
    ax.set_xticks(np.arange(-0.5, n_columns, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(n_rows):
        for j in range(n_columns):
            value = matrix[i, j]
            text = "—" if np.isnan(value) else f"{value:.4f}"
            color = "#222222" if np.isnan(value) or value < 0.535 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color)
    # Visual separators between sparse, existing embedding, and Prompt groups.
    for boundary in (6, 18):
        if boundary < n_columns:
            ax.axvline(boundary - 0.5, color="#333333", linewidth=2.2)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012)
    colorbar.set_label("Accuracy", fontsize=11)
    colorbar.ax.tick_params(labelsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("reports/classification/time_aligned_model_matrix.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/classification/time_aligned_model_matrix_accuracy.png"))
    args = parser.parse_args()
    plot(args.csv, args.output)
    print(args.output)


if __name__ == "__main__":
    main()