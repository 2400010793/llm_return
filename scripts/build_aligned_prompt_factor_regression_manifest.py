"""Build leakage-safe regression tasks for aligned scalar/vector factors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.aligned_prompts import load_aligned_prompt_config


FACTORS = ("direction", "high_deviation", "low_deviation", "magnitude", "clarity", "neutral_curvature")
DEFAULT_TARGETS = {
    "horizon": ("forward_compounded_return_1d", "forward_compounded_return_3d", "forward_compounded_return_5d"),
    "valuation": ("market_adjusted_forward_return_20d",),
    "certainty": ("post_realized_volatility_5d",),
    "volatility": ("post_realized_volatility_5d", "volatility_jump"),
    "impact": ("abnormal_event_return_3d", "event_abs_return_3d"),
    "liquidity": ("volume_shock", "range_shock"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--factor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", default=None, help="comma-separated targets applied to every axis")
    parser.add_argument("--regressors", default="ridge,elasticnet_sgd,huber_sgd")
    parser.add_argument("--representation", default="body_mean")
    parser.add_argument("--target-map-json", type=Path, default=None)
    args = parser.parse_args()
    config = load_aligned_prompt_config(args.config)
    panel_rows = len(pd.read_parquet(args.panel, columns=["row_index"]))
    panel_columns = set(pd.read_parquet(args.panel, columns=None).columns)
    if args.target_map_json:
        targets_by_axis = {
            axis: tuple(value for value in values if value in panel_columns)
            for axis, values in json.loads(args.target_map_json.read_text(encoding="utf-8")).items()
        }
        missing_targets = {
            axis: values for axis, values in json.loads(args.target_map_json.read_text(encoding="utf-8")).items()
            if any(value not in panel_columns for value in values)
        }
        if missing_targets:
            raise ValueError(f"targets missing from panel: {missing_targets}")
    elif args.targets:
        targets_by_axis = {axis: tuple(value.strip() for value in args.targets.split(",") if value.strip()) for axis in config["axes"]}
    else:
        targets_by_axis = {axis: DEFAULT_TARGETS.get(axis, ("next_day_return",)) for axis in config["axes"]}
    rows = []
    for axis in config["axes"]:
        for factor in FACTORS:
            matrix = args.factor_root / "roberta" / axis / "masked_short" / args.representation / f"{factor}.npy"
            metadata = args.factor_root / "roberta" / axis / "masked_short" / args.representation / "metadata.parquet"
            for target in targets_by_axis[axis]:
                if target not in panel_columns:
                    raise ValueError(f"target {target} missing from panel")
                for regressor in [value.strip() for value in args.regressors.split(",") if value.strip()]:
                    tag = f"{axis}_{factor}_{regressor}_{target}"
                    rows.append({
                        "kind": "regression",
                        "axis": axis,
                        "factor": factor,
                        "model": "roberta",
                        "variant": "masked_short",
                        "representation": args.representation,
                        "target": target,
                        "regressor": regressor,
                        "panel": str(args.panel),
                        "matrix": str(matrix),
                        "metadata": str(metadata),
                        "embed_model": "chinese_roberta",
                        "panel_row_index_column": "row_index",
                        "metadata_row_index_column": "row_index",
                        "expected_rows": panel_rows,
                        "output": str(args.output_root / tag / "result.json"),
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"wrote {len(rows)} tasks to {args.output}")


if __name__ == "__main__":
    main()
