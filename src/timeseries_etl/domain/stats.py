"""Pure statistical helpers."""

import numpy as np


def z_score(series: np.ndarray) -> np.ndarray:
    """Compute z-scores for a series."""
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return np.zeros_like(series)
    return (series - mean) / std
