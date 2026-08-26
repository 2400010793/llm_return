"""Learn a prompt-token gate, freeze it, and export downstream representations.

The auxiliary return head is used only to train/select the gate on an inner
chronological split.  It is then discarded.  Predictor-fit and
predictor-validation representations are exported for a separate CPU stage.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_dynamic_prompt_token_gating import (
    atomic_torch,
    fit_with_validation,
    make_lookup,
    parse_years,
    resolve_device,
    year_selection,
)
from src.data.prompt_token_embeddings import PromptTokenEmbeddingStore
from src.evaluation.artifacts import atomic_json, experiment_id, file_fingerprint
from src.models.dynamic_token_training import (
    TargetTransform,
    encode_token_store,
    token_weight_summary,
)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def validate_windows(args: argparse.Namespace) -> None:
    inner_fit = set(args.fit_years)
    inner_validation = set(args.validation_years)
    predictor_fit = set(args.predictor_fit_years)
    predictor_validation = set(args.predictor_validation_years)
    if inner_fit & inner_validation:
        raise ValueError("inner fit and validation years must be disjoint")
    if predictor_fit & predictor_validation:
        raise ValueError("predictor fit and validation years must be disjoint")
    if not (inner_fit | inner_validation).issubset(predictor_fit):
        raise ValueError("inner gate years must be contained in predictor fit years")
    if max(predictor_fit) >= min(predictor_validation):
        raise ValueError("predictor validation must be strictly later than predictor fit")


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("data/processed/prompt_token_embeddings_v5"))
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--variant", choices=("short", "masked_short", "long", "masked_long"), required=True)
    parser.add_argument("--gate-mode", choices=("uniform", "static", "dynamic"), required=True)
    parser.add_argument("--fit-years", type=parse_years, required=True, help="Inner gate-fit years.")
    parser.add_argument("--validation-years", type=parse_years, required=True, help="Inner epoch-selection years.")
    parser.add_argument("--predictor-fit-years", type=parse_years, required=True)
    parser.add_argument("--predictor-validation-years", type=parse_years, required=True)
    parser.add_argument("--row-index-column", default="row_index")
    parser.add_argument("--stock-column", default="stock_id")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--return-column", default="next_day_return")
    parser.add_argument("--classification-target-column", default="next_day_label")
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--expected-rows", type=int, default=350577)
    parser.add_argument("--expected-token-count", type=int, required=True)
    parser.add_argument("--expected-hidden-size", type=int, required=True)
    parser.add_argument("--gate-hidden-size", type=int, default=64)
    parser.add_argument("--representation-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--min-stocks-per-day", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.task = "regression"
    args.run_mode = "representation-only"
    args.test_years = []
    validate_windows(args)

    device = resolve_device(args.device)
    store = PromptTokenEmbeddingStore(
        args.input_root, model=args.model, variant=args.variant,
        expected_shards=args.expected_shards, expected_rows=args.expected_rows,
        expected_token_count=args.expected_token_count,
        expected_hidden_size=args.expected_hidden_size,
    )
    spec: dict[str, Any] = {
        "format_version": "frozen_prompt_gate_representation_v1",
        "panel": file_fingerprint(args.panel, hash_content=True),
        "embedding": {
            "root": str(args.input_root), "model": args.model,
            "variant": args.variant, "rows": store.rows,
            "token_count": store.token_count, "hidden_size": store.hidden_size,
        },
        "gate": {
            "mode": args.gate_mode, "gate_hidden_size": args.gate_hidden_size,
            "representation_size": args.representation_size,
            "dropout": args.dropout, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "max_epochs": args.max_epochs,
            "patience": args.patience, "seed": args.seed,
        },
        "windows": {
            "inner_fit": args.fit_years,
            "inner_validation": args.validation_years,
            "predictor_fit": args.predictor_fit_years,
            "predictor_validation": args.predictor_validation_years,
        },
        "protocol": "auxiliary_head_train_then_discard_and_freeze_gate",
    }
    spec_id = experiment_id(spec)
    artifact_dir = args.output.with_suffix(".artifacts")
    completed = artifact_dir / "COMPLETED"
    if completed.is_file() and (artifact_dir / "spec.json").is_file():
        prior = json.loads((artifact_dir / "spec.json").read_text(encoding="utf-8"))
        if prior.get("experiment_id") == spec_id:
            print(json.dumps({"output": str(args.output), "resumed": True, "experiment_id": spec_id}))
            return
    atomic_json(artifact_dir / "spec.json", {**spec, "experiment_id": spec_id})

    columns = [
        args.row_index_column, args.stock_column, args.date_column,
        args.return_column, args.classification_target_column,
    ]
    panel = pd.read_parquet(args.panel, columns=columns)
    panel["_model_date"] = pd.to_datetime(panel[args.date_column], errors="coerce")
    if panel[args.row_index_column].duplicated().any():
        raise ValueError("panel row_index must be unique")
    panel_by_row = panel.set_index(args.row_index_column, drop=False).sort_index()
    targets_by_row = make_lookup(
        panel, row_column=args.row_index_column, values=panel[args.return_column],
        max_row_index=store.max_row_index,
    )
    inner_fit = year_selection(
        panel, row_column=args.row_index_column, years=args.fit_years,
        max_row_index=store.max_row_index,
    )
    inner_validation = year_selection(
        panel, row_column=args.row_index_column, years=args.validation_years,
        max_row_index=store.max_row_index,
    )
    predictor_fit = year_selection(
        panel, row_column=args.row_index_column, years=args.predictor_fit_years,
        max_row_index=store.max_row_index,
    )
    predictor_validation = year_selection(
        panel, row_column=args.row_index_column, years=args.predictor_validation_years,
        max_row_index=store.max_row_index,
    )
    transform = TargetTransform.fit(
        targets_by_row[np.flatnonzero(inner_fit)], task="regression",
    )
    model, _, training = fit_with_validation(
        args, store, inner_fit, inner_validation, targets_by_row,
        panel_by_row, transform, device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fit_encoded = encode_token_store(
        model, store, predictor_fit, targets_by_row,
        device=device, batch_size=args.prediction_batch_size,
    )
    validation_encoded = encode_token_store(
        model, store, predictor_validation, targets_by_row,
        device=device, batch_size=args.prediction_batch_size,
    )
    representation_dir = artifact_dir / "representations"
    atomic_npy(representation_dir / "predictor_fit_rows.npy", fit_encoded.row_indexes)
    atomic_npy(representation_dir / "predictor_fit.npy", fit_encoded.representations)
    atomic_npy(representation_dir / "predictor_validation_rows.npy", validation_encoded.row_indexes)
    atomic_npy(representation_dir / "predictor_validation.npy", validation_encoded.representations)
    atomic_json(
        representation_dir / "predictor_fit_weight_summary.json",
        token_weight_summary(fit_encoded.weights, store.prompt_tokens),
    )
    atomic_json(
        representation_dir / "predictor_validation_weight_summary.json",
        token_weight_summary(validation_encoded.weights, store.prompt_tokens),
    )
    atomic_torch(artifact_dir / "gate_checkpoint.pt", {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "target_transform": transform.__dict__, "best_epoch": training["best_epoch"],
        "prediction_head_status": "discarded_for_downstream_stage",
    })
    report = {
        "format_version": spec["format_version"], "experiment_id": spec_id,
        "spec": spec, "artifact_dir": str(artifact_dir),
        "training": training,
        "exports": {
            "predictor_fit_rows": int(len(fit_encoded.row_indexes)),
            "predictor_validation_rows": int(len(validation_encoded.row_indexes)),
            "representation_dimension": int(fit_encoded.representations.shape[1]),
        },
        "runtime": {"seconds": time.perf_counter() - started, "device": str(device)},
        "provenance": {
            "task_record_id": os.environ.get("TASK_RECORD_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(), "torch": torch.__version__,
        },
    }
    atomic_json(args.output, report)
    atomic_json(artifact_dir / "report.json", report)
    completed.write_text(spec_id + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "experiment_id": spec_id,
        "best_epoch": training["best_epoch"], "resumed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
