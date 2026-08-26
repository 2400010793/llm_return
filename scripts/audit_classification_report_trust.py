"""Build a read-only trust audit of historical classification reports."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "classification"
RULES = (
    (
        "time_aligned_prompt_embeddings",
        "invalid_alignment",
        "350,577-row embeddings were indexed by positions from the 57,741-row legacy panel without metadata-key alignment.",
    ),
    (
        "transformers_time_aligned",
        "mislabelled_representation",
        "Transformer-named files contain non-transformer representation labels (for example word_tfidf); filenames are not reliable evidence.",
    ),
    (
        "time_aligned_embedding_extended",
        "valid_legacy",
        "Legacy 98-stock experiment is internally row-aligned; it is valid only for its 57,741-row panel and 3,050-row 2026 test set.",
    ),
    (
        "pooled_embeddings",
        "valid_full_panel",
        "Full-panel pooled pipeline uses row-index alignment with explicit expected-row audits; results are not directly comparable to legacy tests.",
    ),
)


def scalar_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)] or [{}]
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return [metrics]
    return [payload]


def classify(relative: Path) -> tuple[str, str]:
    top = relative.parts[0]
    for directory, status, rationale in RULES:
        if top == directory:
            return status, rationale
    return "pending", "Not covered by the verified trust rules; requires a separate provenance and alignment audit."


def audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    covered = {rule[0] for rule in RULES}
    for directory in sorted(covered):
        base = REPORT_ROOT / directory
        if not base.exists():
            continue
        # Some completed pooled reports retain a trailing carriage return in
        # the filename from an older CRLF manifest. Treat those as JSON too;
        # the producer and Slurm consumer have since been repaired.
        json_paths = [path for path in base.rglob("*") if path.is_file() and path.name.rstrip().endswith(".json")]
        for path in sorted(json_paths):
            relative = path.relative_to(REPORT_ROOT)
            report_name = "/".join(part.rstrip() for part in relative.parts)
            status, rationale = classify(relative)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                rows.append({
                    "report": report_name, "status": "pending",
                    "rationale": f"Unreadable report: {exc}",
                })
                continue
            for result_number, item in enumerate(scalar_metrics(payload), start=1):
                rows.append({
                    "report": report_name,
                    "result_number": result_number,
                    "status": status,
                    "rationale": rationale,
                    "representation": item.get("representation", payload.get("representation")),
                    "classifier": item.get("classifier", payload.get("classifier")),
                    "reducer": item.get("reducer", payload.get("reducer")),
                    "accuracy": item.get("accuracy"),
                    "balanced_accuracy": item.get("balanced_accuracy"),
                    "auc": item.get("auc"),
                    "majority_accuracy": item.get("majority_accuracy"),
                    "accuracy_lift_vs_majority": item.get("accuracy_lift_vs_majority"),
                    "n_test": item.get("n_test", item.get("n")),
                })
    return rows


def main() -> None:
    rows = audit()
    output_csv = ROOT / "references" / "classification_report_trust_audit.csv"
    output_md = ROOT / "references" / "classification_report_trust_audit.md"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "report", "result_number", "status", "rationale", "representation",
        "classifier", "reducer", "accuracy", "balanced_accuracy", "auc",
        "majority_accuracy", "accuracy_lift_vs_majority", "n_test",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["status"] for row in rows)
    trusted = [row for row in rows if row["status"] in {"valid_legacy", "valid_full_panel"}]
    trusted_with_accuracy = [row for row in trusted if isinstance(row.get("accuracy"), (int, float))]
    trusted_with_accuracy.sort(key=lambda row: float(row["accuracy"]), reverse=True)
    lines = [
        "# Classification report trust audit",
        "",
        "This is a read-only audit of existing result JSON files. Historical reports are not modified. "
        "`valid_legacy` and `valid_full_panel` are scientifically distinct test universes and must not be ranked as if they were one sample.",
        "",
        "## Status counts",
        "",
        "| Status | Result rows |",
        "|---|---:|",
    ]
    lines.extend(f"| `{status}` | {count} |" for status, count in sorted(counts.items()))
    lines.extend([
        "",
        "## Trust rules",
        "",
        "| Directory | Status | Basis |",
        "|---|---|---|",
    ])
    lines.extend(f"| `{directory}` | `{status}` | {rationale} |" for directory, status, rationale in RULES)
    lines.extend([
        "",
        "## Trusted result rows by accuracy (top 25)",
        "",
        "| Scope | Report | Representation | Classifier | Accuracy | Majority | N test |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for row in trusted_with_accuracy[:25]:
        lines.append(
            f"| `{row['status']}` | `{row['report']}` | `{row.get('representation') or ''}` | "
            f"`{row.get('classifier') or ''}` | {float(row['accuracy']):.6f} | "
            f"{float(row['majority_accuracy']):.6f} | {row.get('n_test') or ''} |"
            if isinstance(row.get("majority_accuracy"), (int, float))
            else f"| `{row['status']}` | `{row['report']}` | `{row.get('representation') or ''}` | "
                 f"`{row.get('classifier') or ''}` | {float(row['accuracy']):.6f} |  | {row.get('n_test') or ''} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The historical RoBERTa approximately 54% result remains valid for the legacy 98-stock panel.",
        "- Reports marked `invalid_alignment` must not be cited, ranked, or used for model selection.",
        "- Reports marked `mislabelled_representation` require reconstruction from commands and artifacts before use.",
        "- Future precomputed embedding reports must include the `embedding_alignment` audit emitted by the repaired rolling classifier.",
        "",
    ])
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "status_counts": counts, "csv": str(output_csv), "markdown": str(output_md)}, default=dict, indent=2))


if __name__ == "__main__":
    main()
