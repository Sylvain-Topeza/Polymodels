from __future__ import annotations

from typing import Optional
import numpy as np


def normalize_density(density: np.ndarray, grid: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a nonnegative density on a grid so that integral(density) = 1 using trapezoidal rule.

    If the integral is 0 (or numerically tiny), returns NaNs to signal invalid density.
    """
    d = np.asarray(density, dtype=float).copy()
    g = np.asarray(grid, dtype=float)

    d[d < 0] = 0.0  # guard
    area = float(np.trapz(d, g))
    if area <= eps:
        return np.full_like(d, np.nan)
    return d / area


def normalize_rows(matrix: np.ndarray, grid: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each row of a 2D array as a density over `grid`.
    """
    out = np.empty_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        out[i, :] = normalize_density(matrix[i, :], grid, eps=eps)
    return out