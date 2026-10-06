"""Leakage-safe dataset splits and retrieval evaluation."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    """Row indices for the inner fit/dev split and untouched outer test split."""

    fit: np.ndarray
    dev: np.ndarray
    test: np.ndarray

    @property
    def train(self) -> np.ndarray:
        """Return every non-test row, used as the final retrieval database."""
        return np.sort(np.concatenate((self.fit, self.dev)))


@dataclass(frozen=True)
class PrecisionAtKResult:
    """Per-class retrieval precision and its unweighted macro averages."""

    per_class: dict[str, float]
    query_counts: dict[str, int]
    macro_all: float
    macro_defects: float | None


def retrieval_confusion(
    query_labels: np.ndarray,
    neighbor_labels: np.ndarray,
    class_order: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Return row-normalized neighbor-label frequencies for each query class.

    Row ``i`` describes queries with ``class_order[i]`` and column ``j`` is the
    fraction of their retrieved neighbors labeled ``class_order[j]``.
    """
    query_labels = np.asarray(query_labels)
    neighbor_labels = np.asarray(neighbor_labels)
    classes = list(class_order)
    if query_labels.ndim != 1 or neighbor_labels.ndim != 2:
        raise ValueError("Query labels must be 1D and neighbor labels must be 2D")
    if len(query_labels) != len(neighbor_labels):
        raise ValueError("Queries and neighbor rows must have the same length")
    if len(classes) != len(set(classes)):
        raise ValueError("Class order must not contain duplicates")

    matrix = np.zeros((len(classes), len(classes)), dtype=np.float64)
    for row, query_class in enumerate(classes):
        retrieved = neighbor_labels[query_labels == query_class].reshape(-1)
        if not len(retrieved):
            continue
        for column, neighbor_class in enumerate(classes):
            matrix[row, column] = np.mean(retrieved == neighbor_class)

        # A complete class list should account for every retrieved neighbor.
        if not np.isclose(matrix[row].sum(), 1.0):
            raise ValueError("Class order does not cover all neighbor labels")
    return matrix


def make_stratified_split(
    labels: np.ndarray,
    seed: int,
    test_fraction: float = 0.20,
    dev_fraction_of_train: float = 0.20,
) -> SplitIndices:
    """Create repeatable fit/dev/test indices while preserving class ratios."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array")
    if not 0 < test_fraction < 1:
        raise ValueError("Test fraction must be between 0 and 1")
    if not 0 < dev_fraction_of_train < 1:
        raise ValueError("Dev fraction must be between 0 and 1")

    all_indices = np.arange(len(labels), dtype=np.int64)
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    fit_indices, dev_indices = train_test_split(
        train_indices,
        test_size=dev_fraction_of_train,
        random_state=seed,
        stratify=labels[train_indices],
    )

    return SplitIndices(
        fit=np.sort(fit_indices),
        dev=np.sort(dev_indices),
        test=np.sort(test_indices),
    )


def precision_at_k(
    query_labels: np.ndarray,
    neighbor_labels: np.ndarray,
    k: int = 5,
    none_label: str = "none",
) -> PrecisionAtKResult:
    """Average the fraction of matching neighbor labels for each query class."""
    query_labels = np.asarray(query_labels)
    neighbor_labels = np.asarray(neighbor_labels)

    if query_labels.ndim != 1:
        raise ValueError("Query labels must be one-dimensional")
    if neighbor_labels.ndim != 2:
        raise ValueError("Neighbor labels must be two-dimensional")
    if len(query_labels) != len(neighbor_labels):
        raise ValueError("Queries and neighbor rows must have the same length")
    if k <= 0:
        raise ValueError("k must be positive")
    if neighbor_labels.shape[1] < k:
        raise ValueError(f"Need at least {k} neighbors per query")
    if len(query_labels) == 0:
        raise ValueError("At least one query is required")

    matches = neighbor_labels[:, :k] == query_labels[:, np.newaxis]
    per_query_precision = matches.mean(axis=1)

    class_order = list(dict.fromkeys(query_labels.tolist()))
    per_class = {
        label: float(per_query_precision[query_labels == label].mean())
        for label in class_order
    }
    query_counts = {
        label: int(np.count_nonzero(query_labels == label)) for label in class_order
    }
    defect_scores = [score for label, score in per_class.items() if label != none_label]

    return PrecisionAtKResult(
        per_class=per_class,
        query_counts=query_counts,
        macro_all=float(np.mean(list(per_class.values()))),
        macro_defects=float(np.mean(defect_scores)) if defect_scores else None,
    )


def sample_queries_by_class(
    test_indices: np.ndarray,
    labels: np.ndarray,
    cap_per_class: int,
    seed: int,
) -> np.ndarray:
    """Select up to a fixed number of test queries from every class."""
    test_indices = np.asarray(test_indices, dtype=np.int64)
    labels = np.asarray(labels)
    if cap_per_class <= 0:
        raise ValueError("Query cap must be positive")

    rng = np.random.default_rng(seed)
    test_labels = labels[test_indices]
    class_order = list(dict.fromkeys(test_labels.tolist()))
    selected: list[np.ndarray] = []

    for label in class_order:
        candidates = test_indices[test_labels == label]
        if len(candidates) > cap_per_class:
            candidates = rng.choice(candidates, size=cap_per_class, replace=False)
        selected.append(np.sort(candidates))

    return np.concatenate(selected) if selected else np.array([], dtype=np.int64)


def random_neighbor_indices(
    database_size: int,
    query_count: int,
    k: int,
    seed: int,
) -> np.ndarray:
    """Draw distinct random database positions for each query."""
    if database_size < k:
        raise ValueError("Database must contain at least k rows")
    if query_count < 0:
        raise ValueError("Query count must not be negative")
    if k <= 0:
        raise ValueError("k must be positive")

    rng = np.random.default_rng(seed)
    neighbors = np.empty((query_count, k), dtype=np.int64)

    for column in range(k):
        samples = rng.integers(0, database_size, size=query_count)
        if column:
            duplicate = np.any(neighbors[:, :column] == samples[:, np.newaxis], axis=1)
            while duplicate.any():
                samples[duplicate] = rng.integers(0, database_size, size=duplicate.sum())
                duplicate = np.any(
                    neighbors[:, :column] == samples[:, np.newaxis], axis=1
                )
        neighbors[:, column] = samples

    return neighbors
