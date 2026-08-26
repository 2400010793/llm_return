"""Build matched regression tasks for no-neutral-marker anchor matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGETS = {
    "valuation": ("market_adjusted_forward_return_20d",),
    "certainty": ("post_realized_volatility_5d",),
    "volatility": ("post_realized_volatility_5d", "volatility_jump"),
    "impact": ("abnormal_event_return_3d",),
    "liquidity": ("volume_shock", "range_shock"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regressors", default="ridge,elasticnet_sgd,huber_sgd")
    parser.add_argument("--representation", default="body_mean")
    parser.add_argument("--target-map-json", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    panel_columns = set(pd.read_parquet(args.panel, columns=None).columns)
    targets = json.loads(args.target_map_json.read_text(encoding="utf-8")) if args.target_map_json else TARGETS
    expected_rows = len(pd.read_parquet(
        args.matrix_root / "roberta" / "valuation_anchor" / "masked_short" / args.representation / "metadata.parquet",
        columns=["row_index"],
    ))
    rows = []
    for axis, item in config["prompts"].items():
        matrix = args.matrix_root / "roberta" / str(item["prompt_id"]) / "masked_short" / args.representation / "matrix.npy"
        metadata = args.matrix_root / "roberta" / str(item["prompt_id"]) / "masked_short" / args.representation / "metadata.parquet"
        for target in targets[axis]:
            if target not in panel_columns:
                raise ValueError(f"target missing from panel: {target}")
            for regressor in [value.strip() for value in args.regressors.split(",") if value.strip()]:
                tag = f"{axis}_anchor_{regressor}_{target}"
                rows.append({
                    "kind": "anchor_regression", "axis": axis, "factor": "anchor",
                    "model": "roberta", "variant": "masked_short", "representation": args.representation,
                    "target": target, "regressor": regressor, "panel": str(args.panel),
                    "matrix": str(matrix), "metadata": str(metadata), "embed_model": "chinese_roberta",
                    "panel_row_index_column": "row_index", "metadata_row_index_column": "row_index",
                    "expected_rows": expected_rows, "output": str(args.output_root / tag / "result.json"),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(json.dumps({"tasks": len(rows), "expected_rows": expected_rows, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
