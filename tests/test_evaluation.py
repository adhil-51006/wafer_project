from collections import Counter

import numpy as np
import pytest

from wafer_retrieval.evaluation import (
    make_stratified_split,
    precision_at_k,
    random_neighbor_indices,
    sample_queries_by_class,
)


def balanced_labels(samples_per_class: int = 100) -> np.ndarray:
    return np.repeat(np.array(["Center", "Donut", "Random"]), samples_per_class)


def test_split_is_disjoint_and_covers_every_row() -> None:
    labels = balanced_labels()
    split = make_stratified_split(labels, seed=42)

    assert np.intersect1d(split.fit, split.dev).size == 0
    assert np.intersect1d(split.fit, split.test).size == 0
    assert np.intersect1d(split.dev, split.test).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((split.fit, split.dev, split.test))),
        np.arange(len(labels)),
    )
    np.testing.assert_array_equal(split.train, np.sort(np.concatenate((split.fit, split.dev))))


def test_split_preserves_each_class_proportion() -> None:
    labels = balanced_labels()
    split = make_stratified_split(labels, seed=42)

    assert Counter(labels[split.fit]) == {"Center": 64, "Donut": 64, "Random": 64}
    assert Counter(labels[split.dev]) == {"Center": 16, "Donut": 16, "Random": 16}
    assert Counter(labels[split.test]) == {"Center": 20, "Donut": 20, "Random": 20}


def test_same_seed_produces_the_same_split() -> None:
    labels = balanced_labels()
    first = make_stratified_split(labels, seed=42)
    second = make_stratified_split(labels, seed=42)

    np.testing.assert_array_equal(first.fit, second.fit)
    np.testing.assert_array_equal(first.dev, second.dev)
    np.testing.assert_array_equal(first.test, second.test)


def test_different_seed_changes_the_selected_rows() -> None:
    labels = balanced_labels()
    first = make_stratified_split(labels, seed=42)
    second = make_stratified_split(labels, seed=123)

    assert not np.array_equal(first.test, second.test)


@pytest.mark.parametrize(
    ("test_fraction", "dev_fraction"),
    [(0.0, 0.2), (1.0, 0.2), (0.2, 0.0), (0.2, 1.0)],
)
def test_split_rejects_invalid_fractions(
    test_fraction: float, dev_fraction: float
) -> None:
    with pytest.raises(ValueError, match="fraction"):
        make_stratified_split(
            balanced_labels(),
            seed=42,
            test_fraction=test_fraction,
            dev_fraction_of_train=dev_fraction,
        )


def test_precision_at_five_oracle_scores_one() -> None:
    query_labels = np.array(["Center", "Donut"])
    neighbor_labels = np.array(
        [
            ["Center"] * 5,
            ["Donut"] * 5,
        ]
    )

    result = precision_at_k(query_labels, neighbor_labels)

    assert result.per_class == {"Center": 1.0, "Donut": 1.0}
    assert result.macro_all == 1.0


def test_precision_at_five_always_wrong_scores_zero() -> None:
    query_labels = np.array(["Center", "Donut"])
    neighbor_labels = np.array(
        [
            ["Donut"] * 5,
            ["Center"] * 5,
        ]
    )

    result = precision_at_k(query_labels, neighbor_labels)

    assert result.per_class == {"Center": 0.0, "Donut": 0.0}
    assert result.macro_all == 0.0


def test_precision_at_five_matches_hand_calculated_mixed_case() -> None:
    query_labels = np.array(["Center", "Center", "Donut", "none"])
    neighbor_labels = np.array(
        [
            ["Center", "Center", "Donut", "Center", "none"],
            ["Center", "Donut", "Donut", "none", "Random"],
            ["Donut", "Donut", "Center", "Donut", "Donut"],
            ["none", "Center", "none", "none", "Donut"],
        ]
    )

    result = precision_at_k(query_labels, neighbor_labels)

    assert result.per_class["Center"] == pytest.approx(0.4)
    assert result.per_class["Donut"] == pytest.approx(0.8)
    assert result.per_class["none"] == pytest.approx(0.6)
    assert result.query_counts == {"Center": 2, "Donut": 1, "none": 1}
    assert result.macro_all == pytest.approx(0.6)
    assert result.macro_defects == pytest.approx(0.6)


def test_precision_at_k_rejects_too_few_neighbors() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        precision_at_k(
            np.array(["Center"]),
            np.array([["Center", "Center"]]),
            k=5,
        )


def test_query_sampling_caps_each_class_and_is_repeatable() -> None:
    labels = np.repeat(np.array(["Center", "Donut", "Random"]), [20, 8, 12])
    test_indices = np.arange(len(labels))

    first = sample_queries_by_class(test_indices, labels, cap_per_class=10, seed=42)
    second = sample_queries_by_class(test_indices, labels, cap_per_class=10, seed=42)

    assert Counter(labels[first]) == {"Center": 10, "Donut": 8, "Random": 10}
    np.testing.assert_array_equal(first, second)
    assert set(first) <= set(test_indices)


def test_selected_queries_never_appear_in_train_database() -> None:
    labels = balanced_labels()
    split = make_stratified_split(labels, seed=42)
    queries = sample_queries_by_class(split.test, labels, cap_per_class=10, seed=42)

    assert np.intersect1d(queries, split.train).size == 0


def test_random_neighbors_are_distinct_valid_and_repeatable() -> None:
    first = random_neighbor_indices(database_size=100, query_count=50, k=5, seed=42)
    second = random_neighbor_indices(database_size=100, query_count=50, k=5, seed=42)

    assert first.shape == (50, 5)
    assert first.min() >= 0
    assert first.max() < 100
    assert all(len(set(row)) == 5 for row in first.tolist())
    np.testing.assert_array_equal(first, second)


def test_random_retriever_scores_close_to_database_class_priors() -> None:
    database_labels = np.repeat(np.array(["Center", "Donut", "Random"]), [500, 300, 200])
    query_labels = np.repeat(np.array(["Center", "Donut", "Random"]), 20_000)
    neighbors = random_neighbor_indices(
        database_size=len(database_labels),
        query_count=len(query_labels),
        k=5,
        seed=42,
    )

    result = precision_at_k(query_labels, database_labels[neighbors], k=5)

    assert result.per_class["Center"] == pytest.approx(0.5, abs=0.01)
    assert result.per_class["Donut"] == pytest.approx(0.3, abs=0.01)
    assert result.per_class["Random"] == pytest.approx(0.2, abs=0.01)
