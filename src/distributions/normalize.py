"""
Density normalization on a 1D grid.
"""

from __future__ import annotations

import numpy as np


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration. Equivalent to np.trapezoid / old np.trapz."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    avg_y = 0.5 * (y[:-1] + y[1:])
    return float(np.sum(dx * avg_y))


def normalize_density(density: np.ndarray, grid: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a non-negative density on `grid` so that integral(density) = 1.
    If the integral is too small (numerical zero), returns NaNs.
    """
    d = np.asarray(density, dtype=float).copy()
    d[d < 0] = 0.0
    area = trapezoid(d, grid)
    if area <= eps:
        return np.full_like(d, np.nan)
    return d / area


def normalize_rows(matrix: np.ndarray, grid: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each row of a 2D array as a density on `grid`."""
    out = np.empty_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        out[i, :] = normalize_density(matrix[i, :], grid, eps=eps)
    return out
