import numpy as np

from wafer_retrieval.features import (
    handcrafted_features,
    scale_database_and_queries,
)


def circular_wafer(size: int = 32, radius: float = 14.0) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices((size, size))
    distance = np.hypot(y - (size - 1) / 2, x - (size - 1) / 2)
    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[distance <= radius] = 1
    return wafer, distance


def radial_part(features: np.ndarray) -> np.ndarray:
    return features[:10]


def projection_peaks(features: np.ndarray) -> np.ndarray:
    return features[-8::2]


def test_center_blob_activates_inner_radial_bins() -> None:
    wafer, distance = circular_wafer()
    wafer[distance <= 3] = 2

    radial = radial_part(handcrafted_features(wafer))

    assert int(np.argmax(radial)) <= 2
    assert radial[:3].max() > radial[-3:].max()


def test_edge_ring_activates_outer_radial_bins() -> None:
    wafer, distance = circular_wafer()
    wafer[(distance >= 11) & (distance <= 14)] = 2

    radial = radial_part(handcrafted_features(wafer))

    assert int(np.argmax(radial)) >= 7
    assert radial[-3:].max() > radial[:3].max()


def test_donut_has_low_high_low_radial_profile() -> None:
    wafer, distance = circular_wafer()
    wafer[(distance >= 6) & (distance <= 8)] = 2

    radial = radial_part(handcrafted_features(wafer))
    peak = int(np.argmax(radial))

    assert 3 <= peak <= 6
    assert radial[0] < radial[peak]
    assert radial[-1] < radial[peak]


def test_straight_line_has_stronger_projection_peak_than_scattered_failures() -> None:
    line, _ = circular_wafer()
    line_rows = np.flatnonzero(line[:, 16] == 1)
    line[line_rows, 16] = 2

    scattered, _ = circular_wafer()
    die_positions = np.argwhere(scattered == 1)
    chosen = np.linspace(0, len(die_positions) - 1, len(line_rows), dtype=int)
    scattered_positions = die_positions[chosen]
    scattered[scattered_positions[:, 0], scattered_positions[:, 1]] = 2

    line_peak = projection_peaks(handcrafted_features(line)).max()
    scattered_peak = projection_peaks(handcrafted_features(scattered)).max()

    assert np.count_nonzero(line == 2) == np.count_nonzero(scattered == 2)
    assert line_peak > scattered_peak


def test_empty_and_no_failure_maps_produce_finite_features() -> None:
    empty = np.zeros((32, 32), dtype=np.uint8)
    no_failures, _ = circular_wafer()

    assert np.isfinite(handcrafted_features(empty)).all()
    assert np.isfinite(handcrafted_features(no_failures)).all()


def test_scaler_uses_database_rows_only() -> None:
    features = np.array([[0.0, 10.0], [2.0, 14.0], [1_000.0, -1_000.0]])

    database, queries, scaler = scale_database_and_queries(
        features,
        database_indices=np.array([0, 1]),
        query_indices=np.array([2]),
    )

    np.testing.assert_allclose(scaler.mean_, [1.0, 12.0])
    np.testing.assert_allclose(database.mean(axis=0), [0.0, 0.0], atol=1e-7)
    assert abs(queries[0, 0]) > 100
