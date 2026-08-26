"""Build paired 2024-selection/2025-confirmation two-stage gate matrices."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EMBEDDINGS = (
    ("roberta", "short", 28, 768),
    ("roberta", "masked_short", 28, 768),
    ("bge_m3", "short", 18, 1024),
    ("bge_m3", "masked_short", 18, 1024),
)
GATES = (
    ("uniform64", "uniform", 64, 64, 0.1, 1e-3, 1e-4),
    ("static64", "static", 64, 64, 0.1, 1e-3, 1e-4),
    ("dynamic32", "dynamic", 32, 32, 0.0, 1e-3, 1e-4),
    ("dynamic64", "dynamic", 64, 64, 0.1, 1e-3, 1e-4),
    ("dynamic_rep128", "dynamic", 128, 64, 0.1, 3e-4, 1e-4),
    ("dynamic_wide128", "dynamic", 128, 128, 0.2, 1e-3, 1e-5),
)
WINDOWS = (
    ("tune2024", "2018,2019,2020,2021,2022", "2023", "2018,2019,2020,2021,2022,2023", "2024"),
    ("confirm2025", "2018,2019,2020,2021,2022,2023", "2024", "2018,2019,2020,2021,2022,2023,2024", "2025"),
)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict[str, object]] = []
    for model, variant, token_count, hidden_size in EMBEDDINGS:
        for window, inner_fit, inner_validation, predictor_fit, predictor_validation in WINDOWS:
            for tag, mode, representation_size, gate_hidden_size, dropout, learning_rate, weight_decay in GATES:
                stem = "_".join((model, variant, window, tag, "seed42"))
                report = f"reports/frozen_gate/two_stage_v1/{stem}.json"
                rows.append({
                    "task_id": len(rows), "model": model, "variant": variant,
                    "token_count": token_count, "hidden_size": hidden_size,
                    "window": window, "gate_tag": tag, "gate_mode": mode,
                    "representation_size": representation_size,
                    "gate_hidden_size": gate_hidden_size, "dropout": dropout,
                    "learning_rate": learning_rate, "weight_decay": weight_decay,
                    "inner_fit_years": inner_fit,
                    "inner_validation_years": inner_validation,
                    "predictor_fit_years": predictor_fit,
                    "predictor_validation_years": predictor_validation,
                    "seed": 42, "representation_report": report,
                    "predictor_output": f"reports/frozen_gate/two_stage_v1_predictors/{stem}.json",
                })
    groups = [[], []]
    for position, row in enumerate(rows):
        item = dict(row)
        group = groups[position % 2]
        item["task_id"] = len(group)
        group.append(item)
    write_tsv(Path("configs/generated/two_stage_gate_v1_a.tsv"), groups[0])
    write_tsv(Path("configs/generated/two_stage_gate_v1_b.tsv"), groups[1])
    write_tsv(Path("configs/generated/two_stage_gate_v1_predictors.tsv"), [
        {**row, "task_id": index} for index, row in enumerate(rows)
    ])
    summary = {
        "representation_tasks": len(rows),
        "gpu_group_tasks": [len(group) for group in groups],
        "predictor_tasks": len(rows),
        "embedding_text_combinations": len(EMBEDDINGS),
        "gate_configurations": len(GATES),
        "windows": [item[0] for item in WINDOWS],
        "downstream_reducers": ["none", "pca16", "pca32_if_dimension_allows"],
        "downstream_models": ["ridge", "huber_sgd", "small_mlp"],
        "selection_protocol": "select full config on 2024, exact-key confirmation on 2025",
        "test_year_accessed": False,
    }
    Path("configs/generated/two_stage_gate_v1.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
