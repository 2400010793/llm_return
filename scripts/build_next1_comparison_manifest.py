"""Combine frozen next1 baselines and pending strong models for DM/MCS."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASELINES = {
    "masked_body_mlp": Path("reports/classification/pooled_embeddings/next1_core/roberta_masked_short_body_mean_mlp_none_0.predictions.parquet"),
    "masked_body_logistic": Path("reports/classification/pooled_embeddings/next1_core/roberta_masked_short_body_mean_logistic_none_0.predictions.parquet"),
    "validation_selected_body_mlp": Path("reports/classification/pooled_embeddings/next1_core/roberta_short_body_mean_mlp_none_0.predictions.parquet"),
    "short_full_mean_mlp": Path("reports/classification/pooled_embeddings/next1_core/roberta_short_full_mean_mlp_none_0.predictions.parquet"),
    "masked_segment_gate_mlp": Path("reports/classification/pooled_embeddings/next1_segment_gating/roberta_masked_short_title_body_full_concat_logistic_l1_keep2_simple_mlp.predictions.parquet"),
}


def build_rows(strong_manifest: Path) -> list[dict[str, str]]:
    rows = [{"model": name, "path": str(path)} for name, path in BASELINES.items()]
    with strong_manifest.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            name = "__".join((item["classifier"], item["model"], item["variant"], item["feature"]))
            path = str(Path(item["output"]).with_suffix(".predictions.parquet"))
            rows.append({"model": name, "path": path})
    names = [row["model"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("comparison model names must be unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(args.strong_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "path"), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} comparison models to {args.output}")


if __name__ == "__main__":
    main()
