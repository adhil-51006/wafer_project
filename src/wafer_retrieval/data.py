"""Load and clean the WM-811K wafer-map data."""

from collections.abc import Iterator

import numpy as np
from skimage.transform import resize


def _flatten_nested(value: object) -> Iterator[object]:
    """Yield individual items from nested arrays, lists, and tuples."""
    if isinstance(value, np.ndarray):
        for item in value.flat:
            yield from _flatten_nested(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_nested(item)
    else:
        yield value


def unwrap_label(raw_label: object) -> str | None:
    """Convert a nested WM-811K label into a string or ``None``.

    The original pickle stores labels in cells such as ``array([['Center']])``
    and represents unlabeled wafers with empty arrays. A cell containing two
    different labels is treated as invalid rather than choosing one silently.
    """
    labels: list[str] = []

    for item in _flatten_nested(raw_label):
        if item is None:
            continue
        if isinstance(item, (float, np.floating)) and np.isnan(item):
            continue
        if not isinstance(item, (str, np.str_)):
            raise TypeError(f"Expected a string label, got {type(item).__name__}")

        cleaned = str(item).strip()
        if cleaned:
            labels.append(cleaned)

    unique_labels = list(dict.fromkeys(labels))
    if not unique_labels:
        return None
    if len(unique_labels) > 1:
        raise ValueError(f"Found multiple different labels: {unique_labels}")
    return unique_labels[0]


def resize_wafer_map(wafer_map: object, output_size: int = 32) -> np.ndarray:
    """Resize one categorical wafer map without blending its cell values."""
    wafer = np.asarray(wafer_map)

    if wafer.ndim != 2:
        raise ValueError(f"Wafer map must be two-dimensional, got shape {wafer.shape}")
    if wafer.size == 0:
        raise ValueError("Wafer map must not be empty")
    if output_size <= 0:
        raise ValueError("Output size must be positive")
    if not np.isin(wafer, (0, 1, 2)).all():
        invalid_values = np.unique(wafer[~np.isin(wafer, (0, 1, 2))]).tolist()
        raise ValueError(f"Wafer map contains values outside {{0, 1, 2}}: {invalid_values}")

    if wafer.shape == (output_size, output_size):
        return wafer.astype(np.uint8, copy=True)

    # order=0 selects the nearest source cell. Higher orders interpolate values,
    # which would invent categories between pass (1) and fail (2).
    resized = resize(
        wafer,
        (output_size, output_size),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized.astype(np.uint8)


def process_labeled_maps(
    wafer_maps: object,
    raw_labels: object,
    output_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep labeled records and return aligned maps, labels, and source indices."""
    if len(wafer_maps) != len(raw_labels):  # type: ignore[arg-type]
        raise ValueError("Wafer maps and labels must contain the same number of rows")

    clean_labels: list[str] = []
    source_indices: list[int] = []
    for source_index, raw_label in enumerate(raw_labels):  # type: ignore[union-attr]
        label = unwrap_label(raw_label)
        if label is not None:
            clean_labels.append(label)
            source_indices.append(source_index)

    processed_maps = np.empty(
        (len(source_indices), output_size, output_size), dtype=np.uint8
    )
    for output_index, source_index in enumerate(source_indices):
        processed_maps[output_index] = resize_wafer_map(
            wafer_maps[source_index], output_size  # type: ignore[index]
        )

    label_width = max((len(label) for label in clean_labels), default=1)
    return (
        processed_maps,
        np.asarray(clean_labels, dtype=f"<U{label_width}"),
        np.asarray(source_indices, dtype=np.int64),
    )
