"""Combine frozen training-only Prompt PCA features with pooled text segments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import load_pooled_embeddings


FEATURES = {
    "prompt_pca128_title": ("prompt", "title"),
    "prompt_pca128_body": ("prompt", "body"),
    "prompt_pca128_title_body": ("prompt", "title", "body"),
}


def file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--prompt-pca-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--variant", choices=("short", "masked_short"), required=True)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--expected-rows", type=int, default=903665)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.components < 1 or args.batch_size < 1:
        raise ValueError("components and batch-size must be positive")

    pca_summary_path = args.prompt_pca_dir / "summary.json"
    pca_path = args.prompt_pca_dir / f"pca_{args.components}.npy"
    if not pca_summary_path.is_file() or not pca_path.is_file():
        raise FileNotFoundError(f"missing Prompt PCA output under {args.prompt_pca_dir}")
    pca_summary = json.loads(pca_summary_path.read_text(encoding="utf-8"))
    expected_protocol = "PCA and token gate fit once on the earliest six-year fit population"
    if expected_protocol not in str(pca_summary.get("preprocessing_protocol", "")):
        raise ValueError("Prompt PCA is not declared training-only")
    for key, expected in {
        "model": args.model,
        "variant": args.variant,
        "rows": args.expected_rows,
        "components": args.components,
    }.items():
        if pca_summary.get(key) != expected:
            raise ValueError(
                f"Prompt PCA summary mismatch for {key}: "
                f"{pca_summary.get(key)!r} != {expected!r}"
            )

    panel = pd.read_parquet(args.panel, columns=["row_index"])
    panel_rows = pd.to_numeric(panel["row_index"], errors="raise").to_numpy(dtype=np.int64)
    expected_indices = np.arange(1, args.expected_rows + 1, dtype=np.int64)
    if len(panel_rows) != args.expected_rows or not np.array_equal(
        np.sort(panel_rows), expected_indices
    ):
        raise ValueError("panel row_index must cover 1..expected-rows exactly once")

    prompt = np.load(pca_path, mmap_mode="r")
    if prompt.shape != (args.expected_rows, args.components):
        raise ValueError(
            f"Prompt PCA shape mismatch: {prompt.shape} != "
            f"{(args.expected_rows, args.components)}"
        )
    pooled = load_pooled_embeddings(
        args.embedding_root,
        model=args.model,
        variant=args.variant,
        feature="title_body_concat",
        require_complete_rows=args.expected_rows,
        max_matrix_gib=16.0,
    )
    if len(pooled.component_shapes) != 2:
        raise ValueError(f"expected title/body components; found {pooled.component_shapes}")
    title_shape, body_shape = pooled.component_shapes
    if len(title_shape) != 1 or title_shape != body_shape:
        raise ValueError(f"title/body shape mismatch: {pooled.component_shapes}")
    hidden_size = int(title_shape[0])
    if pooled.matrix.shape != (args.expected_rows, hidden_size * 2):
        raise ValueError("pooled title/body matrix has an unexpected shape")
    actual_indices = pooled.metadata["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError("pooled embeddings are not in ascending source row_index order")

    inputs = {
        "panel": file_identity(args.panel),
        "prompt_pca": file_identity(pca_path),
        "prompt_pca_summary": file_identity(pca_summary_path),
        "pooled_parts": [str(path.resolve()) for path in pooled.parts],
    }
    summary_path = args.output_dir / "summary.json"
    completed_path = args.output_dir / "COMPLETED"
    if summary_path.is_file() and completed_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        outputs_exist = all(
            (args.output_dir / f"{name}.npy").is_file() for name in FEATURES
        ) and (args.output_dir / "metadata.parquet").is_file()
        if existing.get("inputs") == inputs and outputs_exist:
            print(json.dumps({**existing, "resumed": True}, ensure_ascii=False))
            return
        raise FileExistsError(f"stale completed output exists: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    output_specs: dict[str, dict[str, object]] = {}
    writers: dict[str, np.memmap] = {}
    try:
        for name, groups in FEATURES.items():
            dimension = args.components + hidden_size * (len(groups) - 1)
            temporary = args.output_dir / f".{name}.{os.getpid()}.tmp.npy"
            temporary_paths.append(temporary)
            writers[name] = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(args.expected_rows, dimension),
            )
            output_specs[name] = {
                "path": str(args.output_dir / f"{name}.npy"),
                "shape": [args.expected_rows, dimension],
                "groups": list(groups),
                "column_order": "prompt_pca, then requested pooled segments",
            }

        for start in range(0, args.expected_rows, args.batch_size):
            stop = min(start + args.batch_size, args.expected_rows)
            prompt_batch = np.asarray(prompt[start:stop], dtype=np.float32)
            title_batch = np.asarray(pooled.matrix[start:stop, :hidden_size], dtype=np.float32)
            body_batch = np.asarray(pooled.matrix[start:stop, hidden_size:], dtype=np.float32)
            group_values = {
                "prompt": prompt_batch,
                "title": title_batch,
                "body": body_batch,
            }
            for name, groups in FEATURES.items():
                writers[name][start:stop] = np.concatenate(
                    [group_values[group] for group in groups], axis=1
                )
        for writer in writers.values():
            writer.flush()
        writers.clear()

        metadata_temp = args.output_dir / f".metadata.{os.getpid()}.tmp.parquet"
        temporary_paths.append(metadata_temp)
        pd.DataFrame({"row_index": expected_indices}).to_parquet(metadata_temp, index=False)
        for name in FEATURES:
            temporary = args.output_dir / f".{name}.{os.getpid()}.tmp.npy"
            os.replace(temporary, args.output_dir / f"{name}.npy")
            temporary_paths.remove(temporary)
        os.replace(metadata_temp, args.output_dir / "metadata.parquet")
        temporary_paths.remove(metadata_temp)

        summary = {
            "format_version": "prompt_pca_segment_fusion_v1",
            "model": args.model,
            "variant": args.variant,
            "rows": args.expected_rows,
            "prompt_components": args.components,
            "prompt_source_dimension": int(pca_summary["input_dimension"]),
            "prompt_explained_variance": float(pca_summary["explained_variance"]),
            "prompt_pca_fit_years": pca_summary["pca_fit_years"],
            "prompt_preprocessing_protocol": pca_summary["preprocessing_protocol"],
            "pooled_hidden_size": hidden_size,
            "row_order": "source row_index ascending, exact 1..rows",
            "inputs": inputs,
            "outputs": output_specs,
        }
        summary_temp = args.output_dir / f".summary.{os.getpid()}.tmp.json"
        temporary_paths.append(summary_temp)
        summary_temp.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(summary_temp, summary_path)
        temporary_paths.remove(summary_temp)
        completed_path.write_text("prompt_pca_segment_fusion_v1\n", encoding="ascii")
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        writers.clear()
        for path in temporary_paths:
            if path.is_file():
                path.unlink()


if __name__ == "__main__":
    main()
