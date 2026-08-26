"""Build model-only manifests for existing BGE-M3 and RoBERTa embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = {
    "bge_m3": "bge_m3.npy",
    "chinese_roberta": "chinese_roberta.npy",
}
INPUTS = (
    "text_plain",
    "text_input_2_short",
    "text_input_3_long",
    "text_input_5_masked_short",
    "text_input_6_masked_long",
)
PHASE_CONFIG = {
    "global_logistic": (("logistic", "none", 0),),
    "global_simple_mlp": (("simple_mlp", "none", 0),),
    "global_hist_tree": (("hist_gradient_boosting", "none", 0),),
    "global_nonlinear": (
        ("simple_mlp", "none", 0),
        ("hist_gradient_boosting", "none", 0),
    ),
    "global_sgd": (("sgd", "none", 0),),
    "global_pca": (("logistic", "pca", 128),),
    "label_comparison": (("logistic", "none", 0), ("simple_mlp", "none", 0)),
    "next1_core": (("logistic", "none", 0), ("simple_mlp", "none", 0)),
    "next1_sgd_sentinel": (("sgd", "none", 0),),
}

TARGET_CONFIG = {
    "event3_to_next1": ("event_return_3d", "next_day_return"),
    "next1_to_next1": ("next_day_return", "next_day_return"),
}


def build_rows(root: Path, phase: str, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if phase == "label_comparison":
        target_modes = TARGET_CONFIG.items()
    elif phase.startswith("next1_") or phase.startswith("global_"):
        target_modes = (("next1_to_next1", TARGET_CONFIG["next1_to_next1"]),)
    else:
        raise ValueError(
            "event3_to_next1 is available only through the explicit "
            "label_comparison ablation phase"
        )
    for model, matrix_name in MODELS.items():
        for input_name in INPUTS:
            if phase == "next1_core" and input_name not in {
                "text_plain", "text_input_2_short", "text_input_5_masked_short",
            }:
                continue
            if phase == "next1_sgd_sentinel" and input_name != "text_input_5_masked_short":
                continue
            directory = root / "data/processed/embeddings/all_inputs" / model / input_name
            matrix = directory / matrix_name
            metadata = directory / "metadata.parquet"
            if not matrix.is_file() or not metadata.is_file():
                raise ValueError(f"missing existing embedding assets in {directory}")
            for target_mode, (train_target, evaluation_target) in target_modes:
                for classifier, reducer, components in PHASE_CONFIG[phase]:
                    target_tag = (
                        f"_{target_mode}"
                        if phase == "label_comparison" or phase.startswith("next1_") else ""
                    )
                    output = Path(
                        f"reports/classification/precomputed_embeddings/{phase}/"
                        f"{model}_{input_name}_{classifier}_{reducer}{target_tag}_seed{seed}.json"
                    )
                    rows.append({
                        "task_id": len(rows),
                        "phase": phase,
                        "embedding_model": model,
                        "input_name": input_name,
                        "matrix": str(matrix.relative_to(root)),
                        "metadata": str(metadata.relative_to(root)),
                        "classifier": classifier,
                        "reducer": reducer,
                        "components": components,
                        "seed": seed,
                        "output": str(output),
                        "train_target": train_target,
                        "evaluation_target": evaluation_target,
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=tuple(PHASE_CONFIG), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = build_rows(root, args.phase, args.seed)
    output = args.output or Path(f"configs/generated/precomputed_{args.phase}_seed{args.seed}.tsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "phase": args.phase,
        "tasks": len(rows),
        "models": list(MODELS),
        "inputs": list(INPUTS),
        "seed": args.seed,
        "target_modes": sorted({str(row["train_target"]) for row in rows}),
        "output": str(output),
        "generates_embeddings": False,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
