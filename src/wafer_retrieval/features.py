"""Handcrafted spatial features for categorical wafer maps."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def feature_names(radial_bins: int = 10, angular_bins: int = 12) -> list[str]:
    """Return feature names in the same order as ``handcrafted_features``."""
    names = [f"radial_density_{index}" for index in range(radial_bins)]
    names.extend(f"angular_density_{index}" for index in range(angular_bins))
    names.extend(
        [
            "fail_density",
            "fail_centroid_y",
            "fail_centroid_x",
            "fail_radius_mean",
            "fail_radius_std",
            "fail_bbox_area_ratio",
        ]
    )
    for direction in ("horizontal", "vertical", "diagonal", "anti_diagonal"):
        names.extend((f"projection_peak_{direction}", f"projection_spread_{direction}"))
    return names


def _region_densities(
    die_mask: np.ndarray,
    fail_mask: np.ndarray,
    region_ids: np.ndarray,
    number_regions: int,
) -> np.ndarray:
    """Calculate failed-die density using only real dies as denominators."""
    densities = np.zeros(number_regions, dtype=np.float32)
    for region in range(number_regions):
        real_dies = die_mask & (region_ids == region)
        denominator = np.count_nonzero(real_dies)
        if denominator:
            densities[region] = np.count_nonzero(fail_mask & real_dies) / denominator
    return densities


def _projection_summaries(fail_mask: np.ndarray) -> np.ndarray:
    """Summarize four discrete Radon-style projections of failed pixels."""
    fail_count = int(np.count_nonzero(fail_mask))
    if fail_count == 0:
        return np.zeros(8, dtype=np.float32)

    height, width = fail_mask.shape
    offsets = range(-(height - 1), width)
    projections = (
        fail_mask.sum(axis=1),
        fail_mask.sum(axis=0),
        np.array([np.diagonal(fail_mask, offset).sum() for offset in offsets]),
        np.array(
            [np.diagonal(np.fliplr(fail_mask), offset).sum() for offset in offsets]
        ),
    )

    summaries: list[float] = []
    for projection in projections:
        summaries.extend(
            (
                float(projection.max() / fail_count),
                float(projection.std() / fail_count),
            )
        )
    return np.asarray(summaries, dtype=np.float32)


def handcrafted_features(
    wafer_map: np.ndarray,
    radial_bins: int = 10,
    angular_bins: int = 12,
) -> np.ndarray:
    """Describe one wafer by failure location, spread, amount, and alignment."""
    wafer = np.asarray(wafer_map)
    if wafer.ndim != 2 or wafer.size == 0:
        raise ValueError("Wafer map must be a non-empty two-dimensional array")
    if not np.isin(wafer, (0, 1, 2)).all():
        raise ValueError("Wafer map contains values outside {0, 1, 2}")
    if radial_bins <= 0 or angular_bins <= 0:
        raise ValueError("Bin counts must be positive")

    die_mask = wafer > 0
    fail_mask = wafer == 2
    height, width = wafer.shape
    y, x = np.indices(wafer.shape, dtype=np.float32)
    center_y = (height - 1) / 2
    center_x = (width - 1) / 2
    relative_y = y - center_y
    relative_x = x - center_x
    radius = np.hypot(relative_y, relative_x)

    if np.any(die_mask):
        wafer_radius = float(radius[die_mask].max())
    else:
        wafer_radius = 1.0
    normalized_radius = radius / max(wafer_radius, 1.0)
    radial_ids = np.minimum(
        (normalized_radius * radial_bins).astype(np.int32), radial_bins - 1
    )
    radial = _region_densities(die_mask, fail_mask, radial_ids, radial_bins)

    angle = (np.arctan2(relative_y, relative_x) + np.pi) / (2 * np.pi)
    angular_ids = np.minimum((angle * angular_bins).astype(np.int32), angular_bins - 1)
    angular = _region_densities(die_mask, fail_mask, angular_ids, angular_bins)

    die_count = int(np.count_nonzero(die_mask))
    fail_count = int(np.count_nonzero(fail_mask))
    global_features = np.zeros(6, dtype=np.float32)
    if die_count:
        global_features[0] = fail_count / die_count
    if fail_count:
        failed_y = relative_y[fail_mask]
        failed_x = relative_x[fail_mask]
        failed_radius = normalized_radius[fail_mask]
        fail_rows, fail_columns = np.nonzero(fail_mask)
        bounding_box_area = (
            (int(fail_rows.max()) - int(fail_rows.min()) + 1)
            * (int(fail_columns.max()) - int(fail_columns.min()) + 1)
        )
        global_features[1] = failed_y.mean() / max(center_y, 1.0)
        global_features[2] = failed_x.mean() / max(center_x, 1.0)
        global_features[3] = failed_radius.mean()
        global_features[4] = failed_radius.std()
        global_features[5] = bounding_box_area / max(die_count, 1)

    features = np.concatenate(
        (radial, angular, global_features, _projection_summaries(fail_mask))
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced NaN or infinity")
    return features


def extract_feature_matrix(
    wafer_maps: np.ndarray,
    radial_bins: int = 10,
    angular_bins: int = 12,
) -> np.ndarray:
    """Extract one handcrafted feature row for each wafer map."""
    number_features = len(feature_names(radial_bins, angular_bins))
    matrix = np.empty((len(wafer_maps), number_features), dtype=np.float32)
    for index, wafer_map in enumerate(wafer_maps):
        matrix[index] = handcrafted_features(wafer_map, radial_bins, angular_bins)
    return matrix


def scale_database_and_queries(
    features: np.ndarray,
    database_indices: np.ndarray,
    query_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a scaler on database rows only, then transform database and queries."""
    scaler = StandardScaler()
    database_features = scaler.fit_transform(features[database_indices]).astype(np.float32)
    query_features = scaler.transform(features[query_indices]).astype(np.float32)
    return database_features, query_features, scaler
