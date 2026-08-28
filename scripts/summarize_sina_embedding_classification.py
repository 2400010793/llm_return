"""Summarize and audit completed Sina embedding classification baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.classification import evaluate_binary_classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/results"
        ),
    )
    parser.add_argument("--expected-panel-rows", type=int, default=4928)
    parser.add_argument("--expected-test-rows", type=int, default=None)
    args = parser.parse_args()

    summary_rows: list[dict] = []
    year_rows: list[dict] = []
    package_audits: list[dict] = []
    report_paths = sorted({
        *args.results_dir.glob("*_none.json"),
        *args.results_dir.glob("*_pca*.json"),
    })
    if not report_paths:
        raise ValueError(f"no baseline reports found in {args.results_dir}")

    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        experiment = report["experiment"]
        prediction_path = report_path.with_suffix(".predictions.parquet")
        predictions = pd.read_parquet(prediction_path)
        metrics = evaluate_binary_classification(
            predictions[report["design"]["evaluation_target"]],
            predictions["probability"],
        )
        summary_rows.append({
            "embedding_model": experiment["embedding_model"],
            "classifier": experiment["classifier"],
            "reducer": experiment["reducer"],
            "reducer_components": (
                experiment["reducer_components"]
                if experiment["reducer"] == "pca"
                else None
            ),
            "folds": len(report["results"]),
            "prediction_rows": len(predictions),
            **metrics,
        })
        for fold in report["results"]:
            year_rows.append({
                "embedding_model": experiment["embedding_model"],
                "classifier": experiment["classifier"],
                "reducer": experiment["reducer"],
                "reducer_components": (
                    experiment["reducer_components"]
                    if experiment["reducer"] == "pca"
                    else None
                ),
                "test_year": fold["test_year"],
                "n": int(fold["n"]),
                "n_fit_rows": fold["n_fit"],
                "n_fit_labeled": fold.get("n_fit_labeled"),
                "n_validation_rows": fold["n_validation"],
                "n_validation_labeled": fold.get("n_validation_labeled"),
                "n_all_train_labeled": fold.get("n_all_train_labeled"),
                "n_test_labeled": fold.get("n_test_labeled", int(fold["n"])),
                "positive_rate": fold["positive_rate"],
                "accuracy": fold["accuracy"],
                "majority_accuracy": fold["majority_accuracy"],
                "accuracy_lift_vs_majority": fold["accuracy_lift_vs_majority"],
                "balanced_accuracy": fold["balanced_accuracy"],
                "auc": fold["auc"],
                "f1": fold["f1"],
                "mcc": fold["mcc"],
                "predicted_positive_rate": fold["predicted_positive_rate"],
                "validation_accuracy": fold["validation_accuracy"],
            })

        artifacts = Path(report["artifact_bundle"])
        probabilities = pd.to_numeric(
            predictions["probability"], errors="coerce"
        ).to_numpy(float)
        package_audits.append({
            "report": str(report_path),
            "predictions": str(prediction_path),
            "artifact_bundle": str(artifacts),
            "completed": (artifacts / "COMPLETED").is_file(),
            "manifest": (artifacts / "manifest.json").is_file(),
            "prediction_rows": len(predictions),
            "unique_article_ids": int(predictions["article_id"].nunique()),
            "finite_probabilities": bool(np.isfinite(probabilities).all()),
            "probabilities_in_unit_interval": bool(
                ((probabilities >= 0) & (probabilities <= 1)).all()
            ),
            "test_years": sorted(
                predictions["test_year"].astype(int).unique().tolist()
            ),
            "alignment_matched_rows": report["input"]["alignment"]["matched_rows"],
            "alignment_key_mode": report["input"]["alignment"]["key_mode"],
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "auc"], ascending=False
    ).reset_index(drop=True)
    by_year = pd.DataFrame(year_rows).sort_values(
        ["test_year", "accuracy"], ascending=[True, False]
    ).reset_index(drop=True)
    expected_years = sorted(by_year["test_year"].astype(int).unique().tolist())
    expected_test_rows = (
        args.expected_test_rows
        if args.expected_test_rows is not None
        else package_audits[0]["prediction_rows"]
    )
    designs = {
        (
            report["design"]["train_target"],
            report["design"]["evaluation_target"],
        )
        for report in (
            json.loads(path.read_text(encoding="utf-8")) for path in report_paths
        )
    }
    if len(designs) != 1:
        raise ValueError(f"result directory mixes target designs: {sorted(designs)}")
    train_target, evaluation_target = next(iter(designs))
    all_passed = all(
        row["completed"]
        and row["manifest"]
        and row["finite_probabilities"]
        and row["probabilities_in_unit_interval"]
        and row["prediction_rows"] == expected_test_rows
        and row["unique_article_ids"] == expected_test_rows
        and row["test_years"] == expected_years
        and row["alignment_matched_rows"] == args.expected_panel_rows
        and row["alignment_key_mode"] == "exact_string"
        for row in package_audits
    )
    best = summary.iloc[0]
    best_years = by_year[
        by_year["embedding_model"].eq(best["embedding_model"])
        & by_year["classifier"].eq(best["classifier"])
        & by_year["reducer"].eq(best["reducer"])
        & by_year["reducer_components"].fillna(-1).eq(
            best["reducer_components"] if pd.notna(best["reducer_components"]) else -1
        )
    ]
    audit = {
        "design": {
            "fit_years": 6,
            "validation_years": 2,
            "test_years": 1,
            "rolling_test_years": expected_years,
            "train_target": train_target,
            "evaluation_target": evaluation_target,
            "alignment_key": "article_id",
        },
        "result_packages": package_audits,
        "all_packages_passed": all_passed,
        "interpretation": {
            "best_pooled_model": (
                f"{best['embedding_model']} + {best['classifier']} + "
                f"{best['reducer']}"
            ),
            "best_pooled_accuracy": float(best["accuracy"]),
            "best_pooled_balanced_accuracy": float(best["balanced_accuracy"]),
            "best_pooled_auc": float(best["auc"]),
            "best_model_years_beating_oracle_year_majority": int(
                best_years["accuracy_lift_vs_majority"].gt(0).sum()
            ),
            "conclusion": (
                "weak directional signal; not stable enough for a strong "
                "effectiveness claim"
            ),
        },
    }

    summary.to_csv(args.results_dir / "baseline_summary.csv", index=False)
    by_year.to_csv(args.results_dir / "baseline_by_year.csv", index=False)
    (args.results_dir / "baseline_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit["interpretation"], ensure_ascii=False, indent=2))
    if not all_passed:
        raise RuntimeError("one or more result-package audits failed")


if __name__ == "__main__":
    main()
