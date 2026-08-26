"""Build validation-only overnight matrices for pooled and Prompt-token embeddings."""

from __future__ import annotations

import csv
import json
from pathlib import Path


POOLED_FEATURES = (
    # The completed v3 pooled archives contain mean representations only;
    # cls/max keys were never persisted and must not enter executable grids.
    "title_mean", "body_mean", "title_body_mean", "full_mean",
    "title_body_concat", "title_body_full_concat",
)
CORE_FINE_FEATURES = ("body_mean", "full_mean", "title_body_full_concat")
EMBEDDINGS = (
    ("roberta", "short"), ("roberta", "masked_short"),
    ("bge_m3", "short"), ("bge_m3", "masked_short"),
)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def pooled_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(
        model: str, variant: str, feature: str, regressor: str,
        reducer: str, components: int, stage: str,
    ) -> None:
        tag = "representation" if stage == "coarse" else "fine"
        stem = "_".join((
            model, variant, feature, regressor, reducer, str(components), stage,
        ))
        rows.append({
            "task_id": len(rows), "family": tag,
            "embedding_root": "data/processed/pooled_embeddings_v3",
            "model": model, "variant": variant, "feature": feature,
            "target": "next_day_return", "regressor": regressor,
            "run_mode": "screen", "reducer": reducer,
            "components": components, "search_stage": stage,
            "alphas": (
                "0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000"
                if stage == "fine" else "0.1,1,10,100,1000,10000"
            ),
            "output": f"reports/regression/overnight_v1/{stem}.json",
        })

    # Broad representation/text-processing screen with stable scalable heads.
    for model, variant in EMBEDDINGS:
        for feature in POOLED_FEATURES:
            for regressor in ("ridge", "huber_sgd"):
                add(model, variant, feature, regressor, "none", 0, "coarse")

    # Fine model/reduction search only on the representations most likely to win.
    for model, variant in EMBEDDINGS:
        for feature in CORE_FINE_FEATURES:
            for regressor in ("ridge", "huber_sgd", "small_mlp"):
                for reducer, components in (("none", 0), ("pca", 128)):
                    add(model, variant, feature, regressor, reducer, components, "fine")
    return rows


def dynamic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    shapes = {"roberta": (28, 768), "bge_m3": (18, 1024)}
    hyperparameters = (
        ("uniform_base", "uniform", 64, 64, 0.1, 1e-3, 1e-4),
        ("static_base", "static", 64, 64, 0.1, 1e-3, 1e-4),
        ("dynamic_32", "dynamic", 32, 32, 0.0, 1e-3, 1e-4),
        ("dynamic_base", "dynamic", 64, 64, 0.1, 1e-3, 1e-4),
        ("dynamic_rep128", "dynamic", 128, 64, 0.1, 3e-4, 1e-4),
        ("dynamic_gate128", "dynamic", 64, 128, 0.1, 3e-4, 1e-4),
        ("dynamic_dropout", "dynamic", 64, 64, 0.2, 3e-4, 1e-5),
        ("dynamic_wide", "dynamic", 128, 128, 0.2, 1e-3, 1e-5),
    )
    windows = (
        ("tune2024", "2018,2019,2020,2021,2022,2023", "2024"),
        ("confirm2025", "2018,2019,2020,2021,2022,2023,2024", "2025"),
    )
    for model, variant in EMBEDDINGS:
        token_count, hidden_size = shapes[model]
        for window, fit_years, validation_years in windows:
            for tag, gate_mode, representation_size, gate_hidden_size, dropout, learning_rate, weight_decay in hyperparameters:
                stem = "_".join((model, variant, window, tag, "seed42"))
                rows.append({
                    "task_id": len(rows), "model": model, "variant": variant,
                    "token_count": token_count, "hidden_size": hidden_size,
                    "window": window, "config_tag": tag, "gate_mode": gate_mode,
                    "representation_size": representation_size,
                    "gate_hidden_size": gate_hidden_size, "dropout": dropout,
                    "learning_rate": learning_rate, "weight_decay": weight_decay,
                    "fit_years": fit_years, "validation_years": validation_years,
                    "seed": 42,
                    "output": f"reports/dynamic_prompt/overnight_v1/{stem}.json",
                })
    return rows


def main() -> None:
    pooled = pooled_rows()
    dynamic = dynamic_rows()
    write_tsv(Path("configs/generated/overnight_pooled_v1.tsv"), pooled)
    # Split the GPU matrix into two single-concurrency queues, one per device.
    groups = [[], []]
    for position, row in enumerate(dynamic):
        group = groups[position % 2]
        item = dict(row)
        item["task_id"] = len(group)
        group.append(item)
    write_tsv(Path("configs/generated/overnight_dynamic_v1_a.tsv"), groups[0])
    write_tsv(Path("configs/generated/overnight_dynamic_v1_b.tsv"), groups[1])
    summary = {
        "pooled_tasks": len(pooled), "dynamic_tasks": len(dynamic),
        "dynamic_group_tasks": [len(group) for group in groups],
        "models": ["ridge", "huber_sgd", "small_mlp", "uniform", "static", "dynamic"],
        "embedding_models": ["roberta", "bge_m3"],
        "text_processing": ["short", "masked_short"],
        "test_year_accessed": False,
    }
    Path("configs/generated/overnight_v1.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
