"""Build the unified time-aligned classification matrix.

The primary cell value is test-set accuracy. A long CSV containing all
available metrics is written alongside the wide Markdown matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOTS = (
    "reports/classification/time_aligned_extended",
    "reports/classification/time_aligned_embedding_extended",
    "reports/classification/time_aligned_prompt_embeddings",
)
CLASSIFIER_ORDER = (
    "logistic", "linear_svm", "nb_svm", "multinomial_nb", "complement_nb",
    "sgd", "random_forest", "mlp", "knn",
)
PROMPT_ORDER = ("plain", "short", "long", "masked_short", "masked_long")


def describe_representation(value: str) -> tuple[str, str, str]:
    """Return (model, prompt, representation family) for a report path."""
    if value in {"char_tfidf", "word_tfidf", "word_count"}:
        return value, "none", "sparse"
    if "all_inputs/" in value:
        parts = Path(value.split("all_inputs/", 1)[1]).parts
        prompt = parts[1]
        for prefix in ("text_input_2_", "text_input_3_", "text_input_5_", "text_input_6_"):
            prompt = prompt.replace(prefix, "")
        return parts[0], prompt, "prompt"
    if "qwen3_embedding_8b" in value:
        return "qwen", "existing", "embedding"
    if "/bge_m3/" in value:
        return "bge_m3", "existing", "embedding"
    if "/chinese_roberta/" in value:
        return "chinese_roberta", "existing", "embedding"
    return value, "unknown", "other"


def display_name(model: str, prompt: str, reducer: str) -> str:
    model_label = {
        "char_tfidf": "Char-TF-IDF", "word_tfidf": "Word-TF-IDF",
        "word_count": "Word-Count", "chinese_roberta": "RoBERTa",
        "bge_m3": "BGE-M3", "qwen": "Qwen",
    }.get(model, model)
    prompt_label = {
        "none": "", "existing": "existing", "plain": "plain",
        "short": "short", "long": "long", "masked_short": "masked-short",
        "masked_long": "masked-long",
    }.get(prompt, prompt)
    reducer_label = "no-reduction" if reducer == "none" else reducer.upper()
    pieces = [model_label]
    if prompt_label:
        pieces.append(prompt_label)
    pieces.append(reducer_label)
    return " + ".join(pieces)


def load_results(root: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Load one deterministic result per model/prompt/reducer/classifier cell."""
    cells: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for relative_root in ROOTS:
        for report_path in sorted((root / relative_root).glob("*.json")):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for result in report.get("results", []):
                if not isinstance(result.get("accuracy"), (int, float)):
                    continue
                model, prompt, _ = describe_representation(str(result.get("representation", "")))
                reducer = str(result.get("reducer") or "none")
                classifier = str(result.get("classifier", ""))
                row = dict(result)
                row["source_report"] = str(report_path)
                cells[(model, prompt, reducer, classifier)] = row
    return cells


def ordered_columns(cells: dict[tuple[str, str, str, str], dict[str, Any]]) -> list[tuple[str, str, str]]:
    columns: list[tuple[str, str, str]] = []
    for model in ("char_tfidf", "word_tfidf", "word_count"):
        for reducer in ("none", "svd"):
            if any(key[:3] == (model, "none", reducer) for key in cells):
                columns.append((model, "none", reducer))
    for model in ("chinese_roberta", "bge_m3", "qwen"):
        for prompt in ("existing", *PROMPT_ORDER):
            for reducer in ("none", "pca"):
                if any(key[:3] == (model, prompt, reducer) for key in cells):
                    columns.append((model, prompt, reducer))
    return columns


def format_value(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value:.4f}"


def write_outputs(root: Path, cells: dict[tuple[str, str, str, str], dict[str, Any]]) -> tuple[Path, Path]:
    columns = ordered_columns(cells)
    classifiers = [name for name in CLASSIFIER_ORDER if any(key[3] == name for key in cells)]
    csv_path = root / "reports/classification/time_aligned_model_matrix.csv"
    md_path = root / "reports/classification/time_aligned_model_matrix.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "prompt_state", "reducer", "classifier", "accuracy",
        "balanced_accuracy", "auc", "log_loss", "brier", "test_year",
        "seed", "source_report",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(cells):
            model, prompt, reducer, classifier = key
            row = cells[key]
            output = {field: row.get(field, "") for field in fields}
            output.update({"model": model, "prompt_state": prompt, "reducer": reducer, "classifier": classifier})
            writer.writerow(output)

    lines = [
        "# Time-aligned model/classifier matrix", "",
        "> 主指标：测试集 Accuracy。训练目标为 `event_return_3d`，评价目标为 `next_day_return`。",
        "> 单元格为 Accuracy；`—` 表示该组合尚未生成。AUC 等完整指标请见同名 CSV。", "",
        "| 分类器 | " + " | ".join(display_name(*column) for column in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for classifier in classifiers:
        values = []
        for model, prompt, reducer in columns:
            result = cells.get((model, prompt, reducer, classifier))
            values.append(format_value(result.get("accuracy") if result else None))
        lines.append("| " + classifier + " | " + " | ".join(values) + " |")
    lines.extend([
        "", "## 列定义", "",
        "- `plain`、`short`、`long`、`masked-short`、`masked-long` 是 Prompt 输入状态。",
        "- `PCA`/`SVD` 表示先在训练窗口拟合降维器，再转换 validation/test。",
        "- `existing` 表示已有 embedding，Prompt 状态不可追溯，不应解释为某个具体 Prompt。",
        "- 表中 Accuracy 为主排序口径；建议同时检查多数类 Accuracy 和 Balanced Accuracy。",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    cells = load_results(args.root)
    csv_path, md_path = write_outputs(args.root, cells)
    print(f"cells={len(cells)}")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()