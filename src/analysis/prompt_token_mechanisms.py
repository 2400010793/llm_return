"""Leakage-safe utilities for contextual prompt-token mechanism analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


TARGET_COLUMNS = ("event_return_3d", "next_day_return")
SEEDS = (13, 42, 2024, 2025, 2026)

# Spans are intentionally semantic phrases rather than tokenizer positions.
# Positions are derived separately for every tokenizer from offset mappings.
SEMANTIC_PHRASES: tuple[tuple[str, str], ...] = (
    ("target_horizon", "目标股票在新闻发布后的下一交易日"),
    ("direction", "上涨还是下跌"),
    ("investor_expectation", "改变投资者预期"),
    ("performance_cashflow", "业绩与现金流"),
    ("orders_investment", "订单与投资"),
    ("financing_equity", "融资与股权变动"),
    ("regulation_litigation", "监管与诉讼"),
    ("operating_risk", "经营风险"),
    ("industry_change", "行业变化"),
    ("company_governance", "公司治理"),
    ("noise_filter", "忽略广告、栏目导航和无关模板"),
    ("future_information", "不得使用新闻发布后的价格、收益或其他未来信息"),
    ("information_only", "只依据新闻中能够获得的信息"),
)
GENERIC_GROUP = "generic_instruction"


@dataclass(frozen=True)
class SemanticTokenMap:
    prompt_text: str
    tokens: tuple[str, ...]
    offsets: tuple[tuple[int, int], ...]
    position_groups: tuple[str, ...]
    group_positions: Mapping[str, tuple[int, ...]]

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(self.group_positions)


def _find_unique(text: str, phrase: str) -> tuple[int, int] | None:
    start = text.find(phrase)
    if start < 0:
        return None
    if text.find(phrase, start + 1) >= 0:
        raise ValueError(f"semantic phrase is not unique in prompt: {phrase}")
    return start, start + len(phrase)


def build_semantic_token_map(
    prompt_text: str,
    tokens: Sequence[str],
    offsets: Sequence[Sequence[int]],
    *,
    semantic_phrases: Sequence[tuple[str, str]] = SEMANTIC_PHRASES,
    require_generic: bool = True,
) -> SemanticTokenMap:
    """Map tokenizer-specific positions to canonical prompt semantic groups."""
    if len(tokens) != len(offsets) or not tokens:
        raise ValueError("tokens and offsets must be non-empty and equally sized")
    normalized_offsets = tuple((int(value[0]), int(value[1])) for value in offsets)
    if any(start < 0 or stop < start or stop > len(prompt_text)
           for start, stop in normalized_offsets):
        raise ValueError("token offsets lie outside the prompt text")

    spans = []
    for group, phrase in semantic_phrases:
        span = _find_unique(prompt_text, phrase)
        if span is not None:
            spans.append((group, *span))
    groups: list[str] = []
    for token, (start, stop) in zip(tokens, normalized_offsets):
        # SentencePiece/byte-BPE tokenizers can emit standalone word-boundary
        # markers whose offsets overlap the following character. They are
        # controls, not part of that character's semantic phrase.
        boundary_only = not str(token).replace("▁", "").replace("Ġ", "").strip()
        matches = [] if boundary_only else [
            group for group, span_start, span_stop in spans
            if stop > span_start and start < span_stop
        ]
        if len(matches) > 1:
            raise ValueError(f"token offset {(start, stop)} overlaps semantic groups {matches}")
        groups.append(matches[0] if matches else GENERIC_GROUP)

    group_positions: dict[str, tuple[int, ...]] = {}
    for group, _, _ in spans:
        positions = tuple(index for index, value in enumerate(groups) if value == group)
        if not positions:
            raise ValueError(f"semantic group has no tokenizer positions: {group}")
        group_positions[group] = positions
    generic = tuple(index for index, value in enumerate(groups) if value == GENERIC_GROUP)
    if require_generic and not generic:
        raise ValueError("prompt mapping requires a non-empty generic control group")
    if generic:
        group_positions[GENERIC_GROUP] = generic
    return SemanticTokenMap(
        prompt_text=str(prompt_text), tokens=tuple(str(token) for token in tokens),
        offsets=normalized_offsets, position_groups=tuple(groups),
        group_positions=group_positions,
    )


def validate_tokenizer_mapping(
    tokenizer: object,
    prompt_text: str,
    stored_input_ids: np.ndarray,
    stored_tokens: Sequence[str],
    *,
    semantic_phrases: Sequence[tuple[str, str]] = SEMANTIC_PHRASES,
    require_generic: bool = True,
) -> SemanticTokenMap:
    encoded = tokenizer(
        prompt_text, add_special_tokens=False, return_offsets_mapping=True,
    )
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    expected = np.asarray(stored_input_ids, dtype=np.int64)
    if not np.array_equal(input_ids, expected):
        raise ValueError("current tokenizer input IDs do not match stored prompt_input_ids")
    converted = tuple(str(value) for value in tokenizer.convert_ids_to_tokens(input_ids))
    if converted != tuple(str(value) for value in stored_tokens):
        raise ValueError("current tokenizer tokens do not match stored prompt_tokens.json")
    return build_semantic_token_map(
        prompt_text,
        converted,
        encoded["offset_mapping"],
        semantic_phrases=semantic_phrases,
        require_generic=require_generic,
    )


def rolling_windows(years: Iterable[int]) -> list[dict[str, object]]:
    """Return the repository-standard six/two/one rolling windows."""
    ordered = sorted({int(year) for year in years})
    if len(ordered) < 9:
        raise ValueError("rolling mechanism analysis needs at least nine years")
    return [
        {
            "test_year": ordered[position],
            "fit_years": tuple(ordered[position - 8:position - 2]),
            "validation_years": tuple(ordered[position - 2:position]),
            "all_train_years": tuple(ordered[position - 8:position]),
        }
        for position in range(8, len(ordered))
    ]


class YearlyTokenMoments:
    """Accumulate year/class token moments without retaining document tensors."""

    def __init__(
        self, years: Sequence[int], token_count: int, hidden_size: int,
        targets: Sequence[str] = TARGET_COLUMNS,
    ) -> None:
        self.years = tuple(int(year) for year in years)
        self.year_to_position = {year: index for index, year in enumerate(self.years)}
        self.targets = tuple(targets)
        shape = (len(self.years), token_count, hidden_size)
        self.sum_all = np.zeros(shape, dtype=np.float64)
        self.sq_all = np.zeros(shape, dtype=np.float64)
        self.count_all = np.zeros(len(self.years), dtype=np.int64)
        self.sum_class = {
            target: np.zeros((2, *shape), dtype=np.float64) for target in self.targets
        }
        self.sq_class = {
            target: np.zeros((2, *shape), dtype=np.float64) for target in self.targets
        }
        self.count_class = {
            target: np.zeros((2, len(self.years)), dtype=np.int64)
            for target in self.targets
        }

    def update(
        self, values: np.ndarray, row_years: np.ndarray,
        target_values: Mapping[str, np.ndarray],
    ) -> None:
        tokens = np.asarray(values, dtype=np.float32)
        years = np.asarray(row_years, dtype=np.int64)
        if tokens.ndim != 3 or len(tokens) != len(years):
            raise ValueError("token values and row years must align")
        for year in np.unique(years):
            if int(year) not in self.year_to_position:
                raise ValueError(f"unexpected year in embedding rows: {year}")
            year_position = self.year_to_position[int(year)]
            year_mask = years == year
            selected = tokens[year_mask]
            self.sum_all[year_position] += selected.sum(axis=0, dtype=np.float64)
            self.sq_all[year_position] += np.square(
                selected, dtype=np.float64,
            ).sum(axis=0)
            self.count_all[year_position] += len(selected)
            for target in self.targets:
                numeric = np.asarray(target_values[target], dtype=float)
                if len(numeric) != len(tokens):
                    raise ValueError(f"target {target} does not align with token rows")
                for label in (0, 1):
                    class_mask = year_mask & np.isfinite(numeric)
                    class_mask &= (numeric > 0) if label else (numeric <= 0)
                    classified = tokens[class_mask]
                    if not len(classified):
                        continue
                    self.sum_class[target][label, year_position] += classified.sum(
                        axis=0, dtype=np.float64,
                    )
                    self.sq_class[target][label, year_position] += np.square(
                        classified, dtype=np.float64,
                    ).sum(axis=0)
                    self.count_class[target][label, year_position] += len(classified)

    def save(self, path: object) -> None:
        arrays: dict[str, np.ndarray] = {
            "years": np.asarray(self.years, dtype=np.int32),
            "sum_all": self.sum_all, "sq_all": self.sq_all,
            "count_all": self.count_all,
        }
        for target in self.targets:
            arrays[f"sum_class__{target}"] = self.sum_class[target]
            arrays[f"sq_class__{target}"] = self.sq_class[target]
            arrays[f"count_class__{target}"] = self.count_class[target]
        np.savez_compressed(path, **arrays)


def aggregate_moments(
    archive: Mapping[str, np.ndarray], years: Sequence[int], *, target: str,
) -> dict[str, np.ndarray | int]:
    stored_years = np.asarray(archive["years"], dtype=int)
    positions = [int(np.flatnonzero(stored_years == int(year))[0]) for year in years]
    result: dict[str, np.ndarray | int] = {
        "sum_all": np.asarray(archive["sum_all"])[positions].sum(axis=0),
        "sq_all": np.asarray(archive["sq_all"])[positions].sum(axis=0),
        "count_all": int(np.asarray(archive["count_all"])[positions].sum()),
    }
    for label, name in ((0, "negative"), (1, "positive")):
        result[f"sum_{name}"] = np.asarray(
            archive[f"sum_class__{target}"]
        )[label, positions].sum(axis=0)
        result[f"sq_{name}"] = np.asarray(
            archive[f"sq_class__{target}"]
        )[label, positions].sum(axis=0)
        result[f"count_{name}"] = int(np.asarray(
            archive[f"count_class__{target}"]
        )[label, positions].sum())
    if min(int(result["count_positive"]), int(result["count_negative"])) < 1:
        raise ValueError("selected years require both target classes")
    return result


def token_metrics_from_moments(
    moments: Mapping[str, np.ndarray | int],
    *, prompt_only: np.ndarray | None = None,
    mask_delta_mean: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    count = int(moments["count_all"])
    mean = np.asarray(moments["sum_all"], dtype=np.float64) / count
    variance = np.maximum(
        np.asarray(moments["sq_all"], dtype=np.float64) / count - mean * mean, 0.0,
    )
    class_values = {}
    for name in ("positive", "negative"):
        class_count = int(moments[f"count_{name}"])
        class_mean = np.asarray(moments[f"sum_{name}"], dtype=np.float64) / class_count
        class_variance = np.maximum(
            np.asarray(moments[f"sq_{name}"], dtype=np.float64) / class_count
            - class_mean * class_mean,
            0.0,
        )
        class_values[name] = (class_mean, class_variance)
    positive_mean, positive_variance = class_values["positive"]
    negative_mean, negative_variance = class_values["negative"]
    label_delta = positive_mean - negative_mean
    fisher = np.mean(
        np.square(label_delta) / (positive_variance + negative_variance + 1e-12),
        axis=1,
    )
    result = {
        "mean": mean, "variance": variance,
        "label_delta": label_delta, "fisher": fisher,
        "variance_score": variance.mean(axis=1),
    }
    if prompt_only is not None:
        baseline = np.asarray(prompt_only, dtype=np.float64)
        if baseline.shape != mean.shape:
            raise ValueError("prompt-only baseline shape does not match token moments")
        displacement = mean - baseline
        denominator = np.linalg.norm(mean, axis=1) * np.linalg.norm(baseline, axis=1)
        cosine = 1.0 - np.divide(
            np.sum(mean * baseline, axis=1), denominator,
            out=np.zeros(len(mean), dtype=np.float64), where=denominator > 0,
        )
        result.update({
            "context_delta": displacement,
            "context_cosine_distance": cosine,
            "context_standardized_l2": np.sqrt(np.mean(
                np.square(displacement) / (variance + 1e-12), axis=1,
            )),
        })
    if mask_delta_mean is not None:
        delta = np.asarray(mask_delta_mean, dtype=np.float64)
        if delta.shape != mean.shape:
            raise ValueError("mask delta shape does not match token moments")
        result["mask_delta"] = delta
        result["mask_delta_l2"] = np.linalg.norm(delta, axis=1)
    return result


def cluster_token_responses(
    response_vectors: Sequence[np.ndarray], *, cluster_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster token positions by cosine response, returning labels and linkage."""
    normalized = []
    for values in response_vectors:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("each token response must be token-by-hidden")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized.append(np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0))
    design = np.concatenate(normalized, axis=1)
    if cluster_count < 2 or cluster_count >= len(design):
        raise ValueError("token cluster count must lie in 2..token_count-1")
    distances = pdist(design, metric="cosine")
    distances[~np.isfinite(distances)] = 1.0
    tree = linkage(distances, method="average")
    labels = fcluster(tree, t=cluster_count, criterion="maxclust").astype(np.int32)
    return labels, tree


def select_stable_kmeans(
    fit_features: np.ndarray,
    validation_features: np.ndarray,
    *,
    k_values: Iterable[int] = range(2, 13),
    seeds: Sequence[int] = SEEDS,
    sample_size: int = 10_000,
) -> tuple[int, list[dict[str, float]]]:
    """Choose k using validation silhouette subject to multi-seed ARI stability."""
    fit = np.asarray(fit_features, dtype=np.float32)
    validation = np.asarray(validation_features, dtype=np.float32)
    if fit.ndim != 2 or validation.ndim != 2 or fit.shape[1] != validation.shape[1]:
        raise ValueError("fit and validation cluster features must be aligned matrices")
    if not len(fit) or len(validation) < 3:
        raise ValueError("cluster selection requires non-empty fit and validation rows")
    rng = np.random.default_rng(42)
    score_positions = (
        rng.choice(len(validation), sample_size, replace=False)
        if len(validation) > sample_size else np.arange(len(validation))
    )
    rows: list[dict[str, float]] = []
    for k in k_values:
        if int(k) < 2 or int(k) >= len(fit):
            continue
        assignments = []
        silhouettes = []
        for seed in seeds:
            model = MiniBatchKMeans(
                n_clusters=int(k), random_state=int(seed), batch_size=2048,
                n_init=3, max_iter=200,
            ).fit(fit)
            labels = model.predict(validation)
            sampled = labels[score_positions]
            silhouettes.append(
                float(silhouette_score(validation[score_positions], sampled))
                if len(np.unique(sampled)) > 1 else -1.0
            )
            assignments.append(labels)
        pairwise = [
            adjusted_rand_score(assignments[left], assignments[right])
            for left in range(len(assignments))
            for right in range(left + 1, len(assignments))
        ]
        rows.append({
            "k": float(k), "silhouette_mean": float(np.mean(silhouettes)),
            "silhouette_std": float(np.std(silhouettes)),
            "ari_mean": float(np.mean(pairwise)) if pairwise else 1.0,
            "ari_min": float(np.min(pairwise)) if pairwise else 1.0,
        })
    if not rows:
        raise ValueError("no valid k candidates")
    stable = [row for row in rows if row["ari_mean"] >= 0.7]
    candidates = stable or rows
    winner = max(candidates, key=lambda row: (row["silhouette_mean"], row["ari_mean"], -row["k"]))
    return int(winner["k"]), rows


def incremental_pca_transform(
    fit: np.ndarray, validation: np.ndarray, *, components: int = 32,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, IncrementalPCA]:
    values = np.asarray(fit, dtype=np.float32)
    other = np.asarray(validation, dtype=np.float32)
    count = min(int(components), values.shape[1], len(values) - 1)
    if count < 1:
        raise ValueError("PCA requires at least two fit rows and one feature")
    effective_batch = max(count, min(int(batch_size), len(values)))
    model = IncrementalPCA(n_components=count, batch_size=effective_batch)
    model.fit(values)
    return (
        model.transform(values).astype(np.float32),
        model.transform(other).astype(np.float32), model,
    )
