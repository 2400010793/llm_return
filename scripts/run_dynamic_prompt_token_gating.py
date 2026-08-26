"""Train a leakage-safe prompt-token gate on frozen sharded embeddings.

Model selection is performed on 2018--2023 fit and 2024--2025 stock-day
validation data. ``screen`` mode never evaluates 2026. ``final-test`` first
selects an epoch on validation, then refits on 2018--2025 for exactly that many
epochs before evaluating the available 2026 observations.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prompt_token_embeddings import (
    PromptTokenEmbeddingStore,
    SelectedPromptTokenEmbeddingStore,
)
from src.evaluation.artifacts import (
    acquire_bundle_lock,
    atomic_json,
    completed_bundle_matches,
    experiment_id,
    file_fingerprint,
    release_bundle_lock,
    write_completed,
)
from src.evaluation.classification import evaluate_binary_classification
from src.models.dynamic_token_gating import TokenVariableSelectionNetwork
from src.models.dynamic_token_training import (
    TargetTransform,
    TokenPrediction,
    predict_token_store,
    seed_torch,
    token_weight_summary,
    train_token_epoch,
)
from src.models.token_gating import TokenGate, fit_streaming_token_gate
from src.models.return_prediction import (
    aggregate_stock_day_predictions,
    evaluate_stock_day_predictions,
)
from src.portfolio import portfolio_metrics, quantile_portfolio


def parse_years(value: str) -> list[int]:
    years = [int(item) for item in value.split(",") if item.strip()]
    if not years or len(set(years)) != len(years):
        raise argparse.ArgumentTypeError("years must be a non-empty unique comma list")
    return years


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


TokenStore = Union[PromptTokenEmbeddingStore, SelectedPromptTokenEmbeddingStore]


def make_model(args: argparse.Namespace, store: TokenStore) -> TokenVariableSelectionNetwork:
    return TokenVariableSelectionNetwork(
        store.hidden_size, store.token_count,
        gate_hidden_size=args.gate_hidden_size,
        representation_size=args.representation_size,
        gate_mode=args.gate_mode,
        dropout=args.dropout,
    )


def make_lookup(
    panel: pd.DataFrame,
    *,
    row_column: str,
    values: pd.Series,
    max_row_index: int,
) -> np.ndarray:
    lookup = np.full(max_row_index + 1, np.nan, dtype=np.float64)
    rows = pd.to_numeric(panel[row_column], errors="raise").to_numpy(dtype=np.int64)
    if (rows < 1).any() or (rows > max_row_index).any():
        raise ValueError("panel row_index lies outside prompt embedding coverage")
    lookup[rows] = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return lookup


def year_selection(
    panel: pd.DataFrame,
    *,
    row_column: str,
    years: list[int],
    max_row_index: int,
) -> np.ndarray:
    selected = np.zeros(max_row_index + 1, dtype=bool)
    dates = pd.to_datetime(panel["_model_date"], errors="coerce")
    mask = dates.dt.year.isin(years).to_numpy()
    rows = pd.to_numeric(panel.loc[mask, row_column], errors="raise").to_numpy(dtype=np.int64)
    selected[rows] = True
    return selected


def evaluate_prediction(
    prediction: TokenPrediction,
    panel_by_row: pd.DataFrame,
    *,
    task: str,
    stock_column: str,
    date_column: str,
    return_column: str,
    historical_mean: float,
    min_stocks_per_day: int,
    threshold: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    announcement = panel_by_row.loc[prediction.row_indexes, [
        stock_column, date_column, return_column,
    ]].copy()
    announcement.insert(0, "row_index", prediction.row_indexes)
    announcement["prediction"] = prediction.predictions
    stock_day = aggregate_stock_day_predictions(
        announcement, prediction.predictions,
        stock_column=stock_column, date_column=date_column,
        target_column=return_column,
    )
    if task == "regression":
        metrics, checked_stock_day = evaluate_stock_day_predictions(
            announcement, prediction.predictions,
            historical_mean=historical_mean,
            stock_column=stock_column, date_column=date_column,
            target_column=return_column,
            min_stocks_per_day=min_stocks_per_day,
        )
        stock_day = checked_stock_day
    else:
        metrics = evaluate_binary_classification(
            stock_day["actual_return"], stock_day["prediction"], threshold=threshold,
        )
        metrics["stock_days"] = float(len(stock_day))
    metrics["announcement_loss"] = float(prediction.mean_loss)
    metrics["announcements"] = float(len(announcement))
    return metrics, announcement.reset_index(drop=True), stock_day


def selection_score(task: str, metrics: dict[str, float]) -> float:
    key = "rank_ic_mean" if task == "regression" else "auc"
    score = float(metrics.get(key, float("nan")))
    return score if np.isfinite(score) else -np.inf


def select_classification_threshold(stock_day: pd.DataFrame) -> tuple[float, dict[str, float]]:
    candidates = np.linspace(0.1, 0.9, 161)
    scored = []
    for threshold in candidates:
        metrics = evaluate_binary_classification(
            stock_day["actual_return"], stock_day["prediction"], threshold=float(threshold),
        )
        scored.append((metrics["balanced_accuracy"], -abs(float(threshold) - 0.5), float(threshold), metrics))
    best = max(scored, key=lambda item: (item[0], item[1]))
    return best[2], best[3]


def prediction_weights_frame(
    prediction: TokenPrediction, panel_by_row: pd.DataFrame,
    *, stock_column: str, date_column: str,
) -> pd.DataFrame:
    identity = panel_by_row.loc[prediction.row_indexes, [stock_column, date_column]].reset_index(drop=True)
    values = pd.DataFrame(
        prediction.weights,
        columns=[f"weight_{position + 1:02d}" for position in range(prediction.weights.shape[1])],
    )
    return pd.concat([
        pd.Series(prediction.row_indexes, name="row_index"), identity, values,
    ], axis=1)


def input_spec(args: argparse.Namespace, store: PromptTokenEmbeddingStore) -> dict[str, Any]:
    return {
        "format_version": "dynamic_prompt_token_gate_v2",
        "panel": file_fingerprint(args.panel, hash_content=True),
        "prompt_shards": [
            {
                "directory": str(shard.directory),
                "summary": file_fingerprint(shard.directory / "summary.json", hash_content=True),
                "metadata": file_fingerprint(shard.directory / "metadata.jsonl", hash_content=True),
                "matrix": file_fingerprint(shard.directory / "prompt_token_embeddings.npy"),
            }
            for shard in store.shards
        ],
        "design": {
            "task": args.task, "run_mode": args.run_mode,
            "fit_years": args.fit_years,
            "validation_years": args.validation_years,
            "test_years": args.test_years,
            "prediction_unit": "stock_day_mean_of_announcement_predictions",
            "return_column": args.return_column,
            "classification_target_column": args.classification_target_column,
            "stock_column": args.stock_column, "date_column": args.date_column,
        },
        "model": {
            "embedding_model": args.model, "variant": args.variant,
            "gate_mode": args.gate_mode,
            "source_token_count": store.token_count,
            "effective_token_count": (
                args.hard_gate_keep
                if args.hard_gate_method != "none"
                else store.token_count
            ),
            "hidden_size": store.hidden_size,
            "hard_gate_method": args.hard_gate_method,
            "hard_gate_keep": (
                args.hard_gate_keep if args.hard_gate_method != "none" else None
            ),
            "gate_hidden_size": args.gate_hidden_size,
            "representation_size": args.representation_size,
            "dropout": args.dropout,
        },
        "optimization": {
            "optimizer": "AdamW", "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "batch_size": args.batch_size,
            "max_epochs": args.max_epochs, "patience": args.patience,
            "gradient_clip_norm": args.gradient_clip_norm, "seed": args.seed,
            "regression_loss": "SmoothL1 on fit-standardized target",
            "classification_loss": "BCEWithLogitsLoss",
        },
    }


def fit_with_validation(
    args: argparse.Namespace,
    store: TokenStore,
    fit_selected: np.ndarray,
    validation_selected: np.ndarray,
    targets_by_row: np.ndarray,
    panel_by_row: pd.DataFrame,
    transform: TargetTransform,
    device: torch.device,
) -> tuple[TokenVariableSelectionNetwork, TokenPrediction, dict[str, Any]]:
    seed_torch(args.seed)
    model = make_model(args, store).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_token_epoch(
            model, store, fit_selected, targets_by_row,
            transform=transform, optimizer=optimizer, device=device,
            batch_size=args.batch_size, seed=args.seed + epoch,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        validation_prediction = predict_token_store(
            model, store, validation_selected, targets_by_row,
            transform=transform, device=device, batch_size=args.prediction_batch_size,
        )
        metrics, _, _ = evaluate_prediction(
            validation_prediction, panel_by_row, task=args.task,
            stock_column=args.stock_column, date_column=args.date_column,
            return_column=args.return_column, historical_mean=transform.mean,
            min_stocks_per_day=args.min_stocks_per_day,
        )
        score = selection_score(args.task, metrics)
        history.append({"epoch": epoch, "train_loss": train_loss, "selection_score": score, "validation_metrics": metrics})
        print(json.dumps({
            "epoch": epoch, "train_loss": train_loss,
            "selection_metric": "rank_ic_mean" if args.task == "regression" else "auc",
            "selection_score": score,
        }, ensure_ascii=False), flush=True)
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = clone_state(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no finite validation score was produced")
    model.load_state_dict(best_state)
    best_prediction = predict_token_store(
        model, store, validation_selected, targets_by_row,
        transform=transform, device=device, batch_size=args.prediction_batch_size,
    )
    return model, best_prediction, {
        "best_epoch": best_epoch, "best_score": best_score,
        "epochs_run": len(history), "history": history,
    }


def refit_fixed_epochs(
    args: argparse.Namespace,
    store: TokenStore,
    selected: np.ndarray,
    targets_by_row: np.ndarray,
    transform: TargetTransform,
    device: torch.device,
    epochs: int,
) -> tuple[TokenVariableSelectionNetwork, list[float]]:
    seed_torch(args.seed)
    model = make_model(args, store).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    losses = []
    for epoch in range(1, epochs + 1):
        losses.append(train_token_epoch(
            model, store, selected, targets_by_row,
            transform=transform, optimizer=optimizer, device=device,
            batch_size=args.batch_size, seed=args.seed + epoch,
            gradient_clip_norm=args.gradient_clip_norm,
        ))
    return model, losses


def fit_hard_prompt_gate(
    store: PromptTokenEmbeddingStore,
    selected_by_row: np.ndarray,
    targets_by_row: np.ndarray,
    *,
    method: str,
    keep_tokens: int,
    batch_size: int,
    seed: int,
) -> TokenGate:
    """Fit one global hard position gate from the requested training window."""

    if method in {"first", "random"}:
        if method == "first":
            selected_positions = np.arange(keep_tokens, dtype=np.int64)
            # Scores make the deterministic ordering explicit in the report.
            scores = -np.arange(store.token_count, dtype=np.float64)
        else:
            rng = np.random.default_rng(seed)
            scores = rng.random(store.token_count)
            ranked = np.lexsort((np.arange(store.token_count), -scores))
            selected_positions = np.sort(ranked[:keep_tokens]).astype(np.int64)
        return TokenGate(
            method=method,
            token_count=store.token_count,
            hidden_size=store.hidden_size,
            selected_positions=selected_positions,
            scores=scores,
        )

    def chunks():
        for batch in store.iter_batches(
            selected_by_row, targets_by_row,
            batch_size=batch_size, shuffle=False, seed=0,
        ):
            yield batch.values.reshape(len(batch.values), -1), batch.targets

    return fit_streaming_token_gate(
        chunks(), token_count=store.token_count,
        keep_tokens=keep_tokens, method=method,
    )


def hard_gate_report(
    gate: TokenGate,
    source_tokens: tuple[str, ...],
    years: list[int],
) -> dict[str, Any]:
    return {
        "method": gate.method,
        "fit_years": years,
        "source_token_count": gate.token_count,
        "keep_tokens": int(len(gate.selected_positions)),
        "selected_positions_zero_based": gate.selected_positions.tolist(),
        "selected_positions_one_based": (gate.selected_positions + 1).tolist(),
        "selected_tokens": [source_tokens[position] for position in gate.selected_positions],
        "selected_scores": gate.scores[gate.selected_positions].tolist(),
        "all_position_scores": gate.scores.tolist(),
        "selection_scope": "training rows only",
    }


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("data/processed/prompt_token_embeddings_v5"))
    parser.add_argument("--model", choices=("roberta", "bge_m3"), default="roberta")
    parser.add_argument("--variant", choices=("short", "masked_short", "long", "masked_long"), default="masked_short")
    parser.add_argument("--task", choices=("regression", "classification"), required=True)
    parser.add_argument("--gate-mode", choices=("uniform", "static", "dynamic"), required=True)
    parser.add_argument(
        "--hard-gate-method",
        choices=("none", "fisher", "variance", "first", "random"),
        default="none",
        help="Optionally retain globally selected training-only prompt positions before the soft gate.",
    )
    parser.add_argument("--hard-gate-keep", type=int, default=4)
    parser.add_argument("--run-mode", choices=("screen", "final-test"), default="screen")
    parser.add_argument("--row-index-column", default="row_index")
    parser.add_argument("--stock-column", default="stock_id")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--return-column", default="next_day_return")
    parser.add_argument("--classification-target-column", default="next_day_label")
    parser.add_argument("--fit-years", type=parse_years, default=parse_years("2018,2019,2020,2021,2022,2023"))
    parser.add_argument("--validation-years", type=parse_years, default=parse_years("2024,2025"))
    parser.add_argument("--test-years", type=parse_years, default=parse_years("2026"))
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--expected-rows", type=int, default=350577)
    parser.add_argument("--expected-token-count", type=int, default=28)
    parser.add_argument("--expected-hidden-size", type=int, default=768)
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
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    if set(args.fit_years) & set(args.validation_years) or set(args.fit_years) & set(args.test_years) or set(args.validation_years) & set(args.test_years):
        raise ValueError("fit, validation, and test years must be disjoint")
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError("max-epochs and patience must be positive")
    if args.hard_gate_keep < 1:
        raise ValueError("hard-gate-keep must be positive")
    if args.run_mode == "screen":
        # The test-year selection remains in the immutable spec but no test row
        # is selected, predicted, or evaluated below.
        pass
    device = resolve_device(args.device)
    source_store = PromptTokenEmbeddingStore(
        args.input_root, model=args.model, variant=args.variant,
        expected_shards=args.expected_shards, expected_rows=args.expected_rows,
        expected_token_count=args.expected_token_count,
        expected_hidden_size=args.expected_hidden_size,
    )
    if args.hard_gate_keep > source_store.token_count:
        raise ValueError(
            f"hard-gate-keep cannot exceed {source_store.token_count} source tokens"
        )
    spec = input_spec(args, source_store)
    spec_id = experiment_id(spec)
    bundle = args.artifact_dir or args.output.with_suffix(".artifacts")
    if not args.force_recompute and completed_bundle_matches(bundle, spec_id):
        print(json.dumps({"output": str(args.output), "experiment_id": spec_id, "resumed": True}))
        return
    writer_lock = acquire_bundle_lock(bundle)
    atexit.register(release_bundle_lock, writer_lock)
    atomic_json(bundle / "spec.json", {**spec, "experiment_id": spec_id})

    required_columns = {
        args.row_index_column, args.stock_column, args.date_column,
        args.return_column, args.classification_target_column,
    }
    panel = pd.read_parquet(args.panel, columns=list(required_columns))
    if panel[args.row_index_column].duplicated().any():
        raise ValueError("panel row_index must be unique")
    panel["_model_date"] = pd.to_datetime(panel[args.date_column], errors="coerce")
    panel_rows = pd.to_numeric(panel[args.row_index_column], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(panel_rows), source_store.row_indexes):
        raise ValueError("panel and prompt-token row_index coverage differ")
    panel_by_row = panel.set_index(args.row_index_column, drop=False).sort_index()
    returns_by_row = make_lookup(
        panel, row_column=args.row_index_column, values=panel[args.return_column],
        max_row_index=source_store.max_row_index,
    )
    loss_values = panel[args.return_column] if args.task == "regression" else panel[args.classification_target_column]
    targets_by_row = make_lookup(
        panel, row_column=args.row_index_column, values=loss_values,
        max_row_index=source_store.max_row_index,
    )
    fit_selected = year_selection(
        panel, row_column=args.row_index_column, years=args.fit_years,
        max_row_index=source_store.max_row_index,
    )
    validation_selected = year_selection(
        panel, row_column=args.row_index_column, years=args.validation_years,
        max_row_index=source_store.max_row_index,
    )
    all_train_selected = fit_selected | validation_selected
    test_selected = year_selection(
        panel, row_column=args.row_index_column, years=args.test_years,
        max_row_index=source_store.max_row_index,
    )
    fit_hard_gate = None
    all_train_hard_gate = None
    fit_store: TokenStore = source_store
    final_store: TokenStore = source_store
    if args.hard_gate_method != "none":
        fit_hard_gate = fit_hard_prompt_gate(
            source_store, fit_selected, targets_by_row,
            method=args.hard_gate_method, keep_tokens=args.hard_gate_keep,
            batch_size=args.prediction_batch_size, seed=args.seed,
        )
        fit_store = source_store.select_positions(fit_hard_gate.selected_positions)
        if args.run_mode == "final-test":
            all_train_hard_gate = fit_hard_prompt_gate(
                source_store, all_train_selected, targets_by_row,
                method=args.hard_gate_method, keep_tokens=args.hard_gate_keep,
                batch_size=args.prediction_batch_size, seed=args.seed,
            )
            final_store = source_store.select_positions(
                all_train_hard_gate.selected_positions,
            )
        else:
            final_store = fit_store
    transform = TargetTransform.fit(targets_by_row[np.flatnonzero(fit_selected)], task=args.task)
    model, validation_prediction, training = fit_with_validation(
        args, fit_store, fit_selected, validation_selected, targets_by_row,
        panel_by_row, transform, device,
    )
    validation_metrics, validation_announcements, validation_stock_day = evaluate_prediction(
        validation_prediction, panel_by_row, task=args.task,
        stock_column=args.stock_column, date_column=args.date_column,
        return_column=args.return_column, historical_mean=transform.mean,
        min_stocks_per_day=args.min_stocks_per_day,
    )
    threshold = 0.5
    threshold_metrics = None
    if args.task == "classification":
        threshold, threshold_metrics = select_classification_threshold(validation_stock_day)
        validation_metrics = evaluate_binary_classification(
            validation_stock_day["actual_return"], validation_stock_day["prediction"],
            threshold=threshold,
        )
        validation_metrics.update({
            "stock_days": float(len(validation_stock_day)),
            "announcement_loss": float(validation_prediction.mean_loss),
            "announcements": float(len(validation_announcements)),
        })
    validation_weight_summary = token_weight_summary(
        validation_prediction.weights, fit_store.prompt_tokens,
    )
    screen_bundle = bundle / "validation"
    atomic_torch(screen_bundle / "checkpoint.pt", {
        "state_dict": clone_state(model), "target_transform": transform.__dict__,
        "best_epoch": training["best_epoch"], "gate_mode": args.gate_mode,
        "hard_gate": (
            hard_gate_report(fit_hard_gate, source_store.prompt_tokens, args.fit_years)
            if fit_hard_gate is not None else None
        ),
    })
    atomic_parquet(screen_bundle / "announcement_predictions.parquet", validation_announcements)
    atomic_parquet(screen_bundle / "stock_day_predictions.parquet", validation_stock_day)
    atomic_parquet(screen_bundle / "token_weights.parquet", prediction_weights_frame(
        validation_prediction, panel_by_row,
        stock_column=args.stock_column, date_column=args.date_column,
    ))
    atomic_json(screen_bundle / "token_weight_summary.json", validation_weight_summary)

    result: dict[str, Any] = {
        "fit_years": args.fit_years, "validation_years": args.validation_years,
        "task": args.task, "gate_mode": args.gate_mode,
        "run_mode": args.run_mode,
        "selection_metric": "rank_ic_mean" if args.task == "regression" else "auc",
        "best_epoch": training["best_epoch"], "best_score": training["best_score"],
        "validation_metrics": validation_metrics,
        "validation_selected_threshold": threshold if args.task == "classification" else None,
        "validation_threshold_metrics": threshold_metrics,
        "target_transform": transform.__dict__,
        "n_fit": fit_store.count_selected(fit_selected, targets_by_row),
        "n_validation": fit_store.count_selected(validation_selected, targets_by_row),
        "validation_token_weight_summary": validation_weight_summary,
        "fit_hard_gate": (
            hard_gate_report(fit_hard_gate, source_store.prompt_tokens, args.fit_years)
            if fit_hard_gate is not None else None
        ),
        "training_history": training["history"],
    }

    if args.run_mode == "final-test":
        all_transform = TargetTransform.fit(
            targets_by_row[np.flatnonzero(all_train_selected)], task=args.task,
        )
        final_model, final_losses = refit_fixed_epochs(
            args, final_store, all_train_selected, targets_by_row, all_transform,
            device, int(training["best_epoch"]),
        )
        test_prediction = predict_token_store(
            final_model, final_store, test_selected, targets_by_row,
            transform=all_transform, device=device,
            batch_size=args.prediction_batch_size,
        )
        test_metrics, test_announcements, test_stock_day = evaluate_prediction(
            test_prediction, panel_by_row, task=args.task,
            stock_column=args.stock_column, date_column=args.date_column,
            return_column=args.return_column, historical_mean=all_transform.mean,
            min_stocks_per_day=args.min_stocks_per_day, threshold=threshold,
        )
        test_weight_summary = token_weight_summary(
            test_prediction.weights, final_store.prompt_tokens,
        )
        test_bundle = bundle / "test"
        atomic_torch(test_bundle / "checkpoint.pt", {
            "state_dict": clone_state(final_model),
            "target_transform": all_transform.__dict__,
            "fit_epochs": training["best_epoch"], "gate_mode": args.gate_mode,
            "hard_gate": (
                hard_gate_report(
                    all_train_hard_gate, source_store.prompt_tokens,
                    args.fit_years + args.validation_years,
                ) if all_train_hard_gate is not None else None
            ),
        })
        atomic_parquet(test_bundle / "announcement_predictions.parquet", test_announcements)
        atomic_parquet(test_bundle / "stock_day_predictions.parquet", test_stock_day)
        atomic_parquet(test_bundle / "token_weights.parquet", prediction_weights_frame(
            test_prediction, panel_by_row,
            stock_column=args.stock_column, date_column=args.date_column,
        ))
        atomic_json(test_bundle / "token_weight_summary.json", test_weight_summary)
        portfolio_summary = None
        if args.task == "regression":
            portfolio = quantile_portfolio(test_stock_day, realized="actual_return")
            atomic_parquet(test_bundle / "quintile_portfolio.parquet", portfolio)
            portfolio_summary = portfolio_metrics(portfolio)
        result.update({
            "test_years": args.test_years, "test_metrics": test_metrics,
            "n_all_train": final_store.count_selected(all_train_selected, targets_by_row),
            "n_test": final_store.count_selected(test_selected, targets_by_row),
            "all_train_target_transform": all_transform.__dict__,
            "final_training_losses": final_losses,
            "test_token_weight_summary": test_weight_summary,
            "all_train_hard_gate": (
                hard_gate_report(
                    all_train_hard_gate, source_store.prompt_tokens,
                    args.fit_years + args.validation_years,
                ) if all_train_hard_gate is not None else None
            ),
            "test_portfolio_metrics": portfolio_summary,
        })

    report = {
        "format_version": "dynamic_prompt_token_gate_report_v2",
        "experiment_id": spec_id, "spec": spec, "result": result,
        "artifacts": str(bundle),
        "runtime": {
            "total_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "device": str(device),
        },
        "provenance": {
            "task_record_id": os.environ.get("TASK_RECORD_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(), "torch": torch.__version__,
        },
        "warnings": [
            "CNINFO announcements are a method replication, not strict news-data replication.",
            "Available 2026 observations are incomplete and previously inspected.",
        ],
    }
    atomic_json(args.output, report)
    atomic_json(bundle / "report.json", report)
    artifact_files = sorted(
        path for path in bundle.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "COMPLETED", ".writer.lock"}
    )
    atomic_json(bundle / "manifest.json", {
        "experiment_id": spec_id,
        "files": {
            str(path.relative_to(bundle)): file_fingerprint(path, hash_content=True)
            for path in artifact_files
        },
        "external_outputs": {
            "report": file_fingerprint(args.output, hash_content=True),
        },
    })
    write_completed(bundle, spec_id)
    release_bundle_lock(writer_lock)
    atexit.unregister(release_bundle_lock)
    print(json.dumps({
        "output": str(args.output), "experiment_id": spec_id,
        "best_epoch": training["best_epoch"], "best_score": training["best_score"],
        "resumed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
