import numpy as np
import pytest

from wafer_retrieval.data import (
    process_labeled_maps,
    resize_wafer_map,
    unwrap_label,
)


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        (np.array([["none"]]), "none"),
        (np.array([["Center"]]), "Center"),
        (np.array([["Donut"]]), "Donut"),
        (np.array([["Edge-Ring"]]), "Edge-Ring"),
        ([["Scratch"]], "Scratch"),
        ("Random", "Random"),
    ],
)
def test_unwrap_label_returns_one_clean_string(raw_label: object, expected: str) -> None:
    assert unwrap_label(raw_label) == expected


@pytest.mark.parametrize(
    "raw_label",
    [
        np.array([]),
        [],
        None,
        np.array([[""]]),
        np.array([[None]], dtype=object),
        np.array([[np.nan]]),
    ],
)
def test_unwrap_label_returns_none_for_missing_labels(raw_label: object) -> None:
    assert unwrap_label(raw_label) is None


def test_unwrap_label_rejects_conflicting_labels() -> None:
    with pytest.raises(ValueError, match="multiple different labels"):
        unwrap_label(np.array([["Center", "Donut"]]))


@pytest.mark.parametrize("input_shape", [(3, 5), (45, 48), (53, 52)])
def test_resize_wafer_map_returns_valid_32_by_32_categories(
    input_shape: tuple[int, int],
) -> None:
    wafer = np.zeros(input_shape, dtype=np.uint8)
    wafer[:, input_shape[1] // 3 : 2 * input_shape[1] // 3] = 1
    wafer[input_shape[0] // 3 : 2 * input_shape[0] // 3, :] = 2

    resized = resize_wafer_map(wafer)

    assert resized.shape == (32, 32)
    assert resized.dtype == np.uint8
    assert set(np.unique(resized)) <= {0, 1, 2}


def test_resize_wafer_map_leaves_32_by_32_input_unchanged() -> None:
    wafer = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) % 3

    resized = resize_wafer_map(wafer)

    np.testing.assert_array_equal(resized, wafer)
    assert resized.dtype == np.uint8


def test_resize_wafer_map_copies_nearest_cells_without_blending() -> None:
    wafer = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 1, 1],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(resize_wafer_map(wafer, output_size=4), expected)


@pytest.mark.parametrize(
    ("wafer", "message"),
    [
        (np.zeros((2, 2, 1), dtype=np.uint8), "two-dimensional"),
        (np.array([[0, 3], [1, 2]], dtype=np.uint8), "values outside"),
        (np.empty((0, 2), dtype=np.uint8), "must not be empty"),
    ],
)
def test_resize_wafer_map_rejects_invalid_maps(
    wafer: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resize_wafer_map(wafer)


def test_process_labeled_maps_keeps_maps_labels_and_indices_aligned() -> None:
    center = np.array([[0, 0], [2, 0]], dtype=np.uint8)
    unlabeled = np.ones((3, 2), dtype=np.uint8)
    random = np.array([[1, 2, 1], [2, 1, 2]], dtype=np.uint8)

    maps, labels, source_indices = process_labeled_maps(
        [center, unlabeled, random],
        [np.array([["Center"]]), np.array([]), np.array([["Random"]])],
        output_size=4,
    )

    assert maps.shape == (2, 4, 4)
    assert labels.tolist() == ["Center", "Random"]
    assert source_indices.tolist() == [0, 2]
    np.testing.assert_array_equal(maps[0], resize_wafer_map(center, 4))
    np.testing.assert_array_equal(maps[1], resize_wafer_map(random, 4))


def test_process_labeled_maps_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same number"):
        process_labeled_maps([np.zeros((2, 2))], [], output_size=4)
