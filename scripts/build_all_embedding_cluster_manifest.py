"""Build the fixed/tuned all-matrix regression manifest."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-root", type=Path, default=ROOT / "matrices")
    p.add_argument("--output", type=Path, default=ROOT / "results_all_embedding_cluster_regression" / "manifest.tsv")
    p.add_argument("--mode", choices=("fixed", "tuned"), help="emit only one execution mode")
    args = p.parse_args()
    rows = []
    skipped = []
    panel = {
        "sina": ROOT.parent / "classification" / "sina_single_stock_classification_panel.parquet",
        "cninfo": Path("data/processed/cninfo_full_classification_panel_2010_2026.parquet"),
    }
    panel_years = {
        name: set(pd.to_datetime(pd.read_parquet(path, columns=["entry_date"])["entry_date"], errors="coerce").dt.year.dropna().astype(int).unique())
        for name, path in panel.items()
    }
    for matrix in sorted(args.matrix_root.glob("*/*/*/*/*/matrix.npy")):
        rel = matrix.relative_to(args.matrix_root)
        dataset, model, prompt, variant, feature = rel.parts[:5]
        metadata = matrix.with_name("metadata.parquet")
        if not metadata.exists() or dataset not in panel:
            continue
        for mode in ((args.mode,) if args.mode else ("fixed", "tuned")):
            for year in range(2018, 2027):
                required = set(range(year - 8, year))
                available = panel_years[dataset]
                if not required.issubset(available):
                    skipped.append({"dataset": dataset, "model": model, "prompt": prompt, "variant": variant, "feature": feature, "test_year": year, "mode": mode, "reason": "strict_6_train_2_validation_history_unavailable", "available_year_min": int(min(available)), "available_year_max": int(max(available))})
                    continue
                rows.append({"dataset": dataset, "model": model, "prompt": prompt, "variant": variant, "feature": feature, "test_year": year, "mode": mode, "panel": str(panel[dataset]), "matrix": str(matrix), "metadata": str(metadata)})
    out = pd.DataFrame(rows).drop_duplicates()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False)
    skipped_path = args.output.with_name(args.output.stem + "_skipped.tsv")
    pd.DataFrame(skipped).to_csv(skipped_path, sep="\t", index=False)
    print(f"wrote {len(out)} rows to {args.output}; skipped {len(skipped)} rows to {skipped_path}")

if __name__ == "__main__":
    main()
