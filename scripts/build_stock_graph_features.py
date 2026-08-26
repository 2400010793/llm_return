"""Build stock-day node features and one-hop graph neighbor aggregates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import align_embeddings_to_panel, load_pooled_embeddings


def aggregate_stock_days(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    *,
    stock_column: str = "stock_id",
    date_column: str = "entry_date",
    target_column: str = "next_day_return",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Average announcement features into unique stock-day nodes."""
    if len(frame) != len(matrix):
        raise ValueError("frame and matrix must contain equal rows")
    work = frame[["row_index", stock_column, date_column, target_column]].copy()
    work[stock_column] = work[stock_column].astype(str).str.extract(
        r"(\d{6})", expand=False
    )
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    if work[[stock_column, date_column]].isna().any().any():
        raise ValueError("stock-day identity contains missing values")
    stock_values = work[stock_column].to_numpy(dtype=str)
    date_values = work[date_column].to_numpy(dtype="datetime64[ns]").view(np.int64)
    order = np.lexsort((stock_values, date_values))
    ordered_dates = date_values[order]
    ordered_stocks = stock_values[order]
    starts = np.r_[
        0,
        1 + np.flatnonzero(
            (ordered_dates[1:] != ordered_dates[:-1])
            | (ordered_stocks[1:] != ordered_stocks[:-1])
        ),
    ]
    counts = np.diff(np.r_[starts, len(order)]).astype(np.int64)
    ordered_matrix = np.asarray(matrix[order], dtype=np.float32)
    aggregated = np.add.reduceat(ordered_matrix, starts, axis=0)
    aggregated /= counts[:, None]

    ordered_targets = pd.to_numeric(
        work[target_column], errors="coerce"
    ).to_numpy(dtype=float)[order]
    finite = np.isfinite(ordered_targets)
    target_sums = np.add.reduceat(np.where(finite, ordered_targets, 0.0), starts)
    target_counts = np.add.reduceat(finite.astype(np.int64), starts)
    targets = np.divide(
        target_sums,
        target_counts,
        out=np.full(len(starts), np.nan, dtype=float),
        where=target_counts > 0,
    )
    target_min = np.minimum.reduceat(
        np.where(finite, ordered_targets, np.inf), starts
    )
    target_max = np.maximum.reduceat(
        np.where(finite, ordered_targets, -np.inf), starts
    )
    inconsistent = (target_counts > 0) & ~np.isclose(
        target_min, target_max, rtol=0.0, atol=1e-12
    )
    if inconsistent.any():
        raise ValueError(f"inconsistent stock-day targets: {int(inconsistent.sum())}")

    ordered_rows = pd.to_numeric(work["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )[order]
    nodes = pd.DataFrame({
        "node_row_index": np.arange(1, len(starts) + 1, dtype=np.int64),
        "entry_date": pd.to_datetime(ordered_dates[starts]),
        "stock_id": ordered_stocks[starts],
        "next_day_return": targets,
        "announcement_count": counts,
        "first_source_row_index": ordered_rows[starts],
    })
    return nodes, np.ascontiguousarray(aggregated, dtype=np.float32)


def shuffled_group_labels(
    stock_to_group: dict[str, str], *, seed: int
) -> dict[str, str]:
    """Permute group membership across stocks while preserving group sizes."""
    stocks = np.array(sorted(stock_to_group), dtype=object)
    groups = np.array([stock_to_group[str(stock)] for stock in stocks], dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = groups[rng.permutation(len(groups))]
    return {str(stock): str(group) for stock, group in zip(stocks, shuffled)}


def graph_group_codes(
    nodes: pd.DataFrame, stock_to_group: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    groups = nodes["stock_id"].map(stock_to_group)
    if groups.isna().any():
        missing = nodes.loc[groups.isna(), "stock_id"].drop_duplicates().head().tolist()
        raise ValueError(f"stocks missing graph groups: {missing}")
    keys = pd.MultiIndex.from_arrays([
        pd.to_datetime(nodes["entry_date"]).to_numpy(dtype="datetime64[ns]"),
        groups.astype(str).to_numpy(),
    ])
    codes, _uniques = pd.factorize(keys, sort=False)
    counts = np.bincount(codes).astype(np.int64)
    return codes.astype(np.int64, copy=False), counts[codes] - 1


def write_neighbor_means(
    features: np.ndarray,
    codes: np.ndarray,
    output: Path,
    *,
    batch_size: int,
) -> np.ndarray:
    """Write mean features of other active nodes in each date/group."""
    if len(features) != len(codes):
        raise ValueError("features and graph codes must contain equal rows")
    order = np.argsort(codes, kind="stable")
    ordered_codes = codes[order]
    starts = np.r_[0, 1 + np.flatnonzero(ordered_codes[1:] != ordered_codes[:-1])]
    unique_codes = ordered_codes[starts]
    sums = np.add.reduceat(np.asarray(features[order], dtype=np.float32), starts, axis=0)
    counts = np.diff(np.r_[starts, len(order)]).astype(np.int64)
    group_sums = np.zeros((int(codes.max()) + 1, features.shape[1]), dtype=np.float32)
    group_counts = np.zeros(int(codes.max()) + 1, dtype=np.int64)
    group_sums[unique_codes] = sums
    group_counts[unique_codes] = counts

    writer = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=features.shape,
    )
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        active_codes = codes[start:stop]
        denominators = group_counts[active_codes] - 1
        values = group_sums[active_codes] - np.asarray(features[start:stop])
        valid = denominators > 0
        values[valid] /= denominators[valid, None]
        values[~valid] = 0.0
        writer[start:stop] = values
    writer.flush()
    del writer
    return group_counts[codes] - 1


def stat_identity(path: Path) -> dict[str, int | str]:
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
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), default="roberta")
    parser.add_argument("--variant", choices=("short", "masked_short"), default="masked_short")
    parser.add_argument("--feature", default="title_body_concat")
    parser.add_argument("--components", type=int, default=256)
    parser.add_argument("--fit-sample-rows", type=int, default=10000)
    parser.add_argument("--expected-rows", type=int, default=903665)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.components, args.fit_sample_rows, args.expected_rows, args.batch_size) < 1:
        raise ValueError("numeric size arguments must be positive")

    summary_path = args.output_dir / "summary.json"
    completed_path = args.output_dir / "COMPLETED"
    if summary_path.is_file() and completed_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        required = ("self_pca256.npy", "industry_neighbor.npy", "random_neighbor.npy", "metadata.parquet")
        if all((args.output_dir / name).is_file() for name in required):
            print(json.dumps({**summary, "resumed": True}, ensure_ascii=False))
            return
        raise FileExistsError(f"incomplete completed output: {args.output_dir}")

    panel = pd.read_parquet(args.panel)
    if len(panel) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} panel rows; found {len(panel)}")
    embeddings = load_pooled_embeddings(
        args.embedding_root,
        model=args.model,
        variant=args.variant,
        feature=args.feature,
        require_complete_rows=args.expected_rows,
        max_matrix_gib=16.0,
    )
    aligned, matrix = align_embeddings_to_panel(panel, embeddings)
    if len(aligned) != args.expected_rows:
        raise ValueError("pooled embeddings do not cover the complete panel")
    nodes, raw_nodes = aggregate_stock_days(aligned, matrix)
    del matrix
    years = pd.to_datetime(nodes["entry_date"]).dt.year.to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero(np.isin(years, np.arange(2010, 2016)))
    if len(fit_rows) < args.components:
        raise ValueError("initial six-year node population is too small for PCA")
    sample_size = min(args.fit_sample_rows, len(fit_rows))
    sample_positions = np.linspace(0, len(fit_rows) - 1, sample_size, dtype=np.int64)
    sampled_rows = fit_rows[sample_positions]
    pca = PCA(
        n_components=args.components,
        svd_solver="randomized",
        iterated_power=3,
        n_oversamples=10,
        random_state=args.seed,
    )
    pca.fit(raw_nodes[sampled_rows])

    universe = pd.read_csv(args.universe, dtype={"stock_id": str})
    required_columns = {"stock_id", "industry"}
    if not required_columns.issubset(universe.columns):
        raise ValueError(f"universe must contain {sorted(required_columns)}")
    universe["stock_id"] = universe["stock_id"].astype(str).str.zfill(6)
    universe = universe.drop_duplicates("stock_id", keep="last")
    if universe["industry"].isna().any():
        universe["industry"] = universe["industry"].fillna("UNKNOWN")
    stock_to_industry = dict(zip(universe["stock_id"], universe["industry"].astype(str)))
    node_stocks = set(nodes["stock_id"])
    if missing := sorted(node_stocks.difference(stock_to_industry)):
        raise ValueError(f"industry coverage missing {len(missing)} stocks: {missing[:10]}")
    random_groups = shuffled_group_labels(stock_to_industry, seed=args.seed)
    industry_codes, industry_neighbor_count = graph_group_codes(nodes, stock_to_industry)
    random_codes, random_neighbor_count = graph_group_codes(nodes, random_groups)
    nodes["industry"] = nodes["stock_id"].map(stock_to_industry)
    nodes["industry_neighbor_count"] = industry_neighbor_count
    nodes["random_neighbor_count"] = random_neighbor_count

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    try:
        self_temp = args.output_dir / f".self_pca256.{os.getpid()}.tmp.npy"
        industry_temp = args.output_dir / f".industry_neighbor.{os.getpid()}.tmp.npy"
        random_temp = args.output_dir / f".random_neighbor.{os.getpid()}.tmp.npy"
        metadata_temp = args.output_dir / f".metadata.{os.getpid()}.tmp.parquet"
        pca_temp = args.output_dir / f".pca256.{os.getpid()}.tmp.joblib"
        temporary.extend([self_temp, industry_temp, random_temp, metadata_temp, pca_temp])
        self_writer = np.lib.format.open_memmap(
            self_temp,
            mode="w+",
            dtype=np.float32,
            shape=(len(nodes), args.components),
        )
        for start in range(0, len(nodes), args.batch_size):
            stop = min(start + args.batch_size, len(nodes))
            self_writer[start:stop] = pca.transform(raw_nodes[start:stop]).astype(np.float32)
        self_writer.flush()
        del self_writer, raw_nodes
        self_features = np.load(self_temp, mmap_mode="r")
        written_industry_count = write_neighbor_means(
            self_features, industry_codes, industry_temp, batch_size=args.batch_size
        )
        written_random_count = write_neighbor_means(
            self_features, random_codes, random_temp, batch_size=args.batch_size
        )
        if not np.array_equal(written_industry_count, industry_neighbor_count):
            raise RuntimeError("industry neighbor counts changed during materialization")
        if not np.array_equal(written_random_count, random_neighbor_count):
            raise RuntimeError("random neighbor counts changed during materialization")
        nodes.to_parquet(metadata_temp, index=False)
        joblib.dump(pca, pca_temp)

        destinations = {
            self_temp: args.output_dir / "self_pca256.npy",
            industry_temp: args.output_dir / "industry_neighbor.npy",
            random_temp: args.output_dir / "random_neighbor.npy",
            metadata_temp: args.output_dir / "metadata.parquet",
            pca_temp: args.output_dir / "pca256.joblib",
        }
        for source, destination in destinations.items():
            os.replace(source, destination)
            temporary.remove(source)

        industry_sizes = universe.loc[
            universe["stock_id"].isin(node_stocks), "industry"
        ].astype(str).value_counts()
        summary = {
            "format_version": "stock_day_graph_features_v1",
            "panel": stat_identity(args.panel),
            "embedding_root": str(args.embedding_root.resolve()),
            "embedding_model": args.model,
            "variant": args.variant,
            "source_feature": args.feature,
            "announcement_rows": int(len(panel)),
            "node_rows": int(len(nodes)),
            "stocks": int(nodes["stock_id"].nunique()),
            "dates": int(nodes["entry_date"].nunique()),
            "date_min": str(nodes["entry_date"].min().date()),
            "date_max": str(nodes["entry_date"].max().date()),
            "pca": {
                "components": args.components,
                "fit_years": list(range(2010, 2016)),
                "fit_population_rows": int(len(fit_rows)),
                "fit_sample_rows": int(sample_size),
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
                "protocol": "fit once on earliest six-year stock-day nodes and freeze for every fold",
            },
            "industry_graph": {
                "source": stat_identity(args.universe),
                "point_in_time": False,
                "status": "exploratory until historical point-in-time industry labels are supplied",
                "groups": int(len(industry_sizes)),
                "group_sizes": {str(k): int(v) for k, v in industry_sizes.items()},
                "nodes_with_active_neighbor": int((industry_neighbor_count > 0).sum()),
                "coverage": float((industry_neighbor_count > 0).mean()),
                "aggregation": "same-entry-date mean of other active stocks in the same industry",
            },
            "random_graph": {
                "seed": args.seed,
                "group_size_policy": "permute industry labels across stocks, preserving group sizes",
                "nodes_with_active_neighbor": int((random_neighbor_count > 0).sum()),
                "coverage": float((random_neighbor_count > 0).mean()),
            },
            "outputs": {
                "self": "self_pca256.npy",
                "industry_neighbor": "industry_neighbor.npy",
                "random_neighbor": "random_neighbor.npy",
                "metadata": "metadata.parquet",
                "pca": "pca256.joblib",
            },
        }
        summary_temp = args.output_dir / f".summary.{os.getpid()}.tmp.json"
        temporary.append(summary_temp)
        summary_temp.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(summary_temp, summary_path)
        temporary.remove(summary_temp)
        completed_path.write_text("stock_day_graph_features_v1\n", encoding="ascii")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        for path in temporary:
            if path.is_file():
                path.unlink()


if __name__ == "__main__":
    main()
