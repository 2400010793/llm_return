"""Build fixed/tuned manifest using full CNINFO pooled embeddings and Sina matrices."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

PROMPT_ROOT = Path("/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1")
POOLED_ROOT = Path("data/processed/pooled_embeddings_2010_2026_clean_v1")
CNINFO_FEATURES = ("prompt_mean", "full_mean", "title_body_mean", "title_body_full_concat")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("fixed", "tuned"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=("sina", "cninfo", "all"), default="all")
    args = p.parse_args()
    rows, skipped = [], []
    sina_panel = PROMPT_ROOT.parent / "classification" / "sina_single_stock_classification_panel.parquet"
    cninfo_panel = Path("data/processed/cninfo_full_classification_panel_2010_2026.parquet")
    def add(item):
        rows.append({"source": item[0], "dataset": item[1], "model": item[2], "prompt": item[3], "variant": item[4], "feature": item[5], "test_year": item[6], "mode": args.mode, "panel": str(item[7]), "matrix": str(item[8]) if item[8] else "-", "metadata": str(item[9]) if item[9] else "-", "embedding_root": str(item[10]) if item[10] else "-", "pooled_feature": item[11] or "-"})
    # Sina: retain every existing prompt-specific matrix, all years are covered.
    if args.dataset in ("sina", "all"):
      for matrix in sorted((PROMPT_ROOT / "matrices" / "sina").glob("*/*/*/*/matrix.npy")):
        rel = matrix.relative_to(PROMPT_ROOT / "matrices" / "sina")
        model, prompt, variant, feature = rel.parts[:4]
        metadata = matrix.with_name("metadata.parquet")
        for year in range(2018, 2027):
            add(("matrix", "sina", model, prompt, variant, feature, year, sina_panel, matrix, metadata, None, None))
    # CNINFO: use the already complete 903,665-row pooled root.
    if args.dataset in ("cninfo", "all"):
      for model in ("roberta", "bge_m3"):
        for variant in ("short", "masked_short"):
            for feature in CNINFO_FEATURES:
                for year in range(2018, 2027):
                    add(("pooled", "cninfo", model, "pooled", variant, feature, year, cninfo_panel, None, None, POOLED_ROOT, feature))
    out = pd.DataFrame(rows).drop_duplicates()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False)
    pd.DataFrame(skipped, columns=["reason"]).to_csv(args.output.with_name(args.output.stem + "_skipped.tsv"), sep="\t", index=False)
    print(f"wrote {len(out)} rows to {args.output}")

if __name__ == "__main__":
    main()
