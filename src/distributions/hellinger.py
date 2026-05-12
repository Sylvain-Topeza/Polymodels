"""
Discrete Hellinger distance between two densities on a common grid.

Reference: Barrau & Douady (2022) ch. 4 §4.3.2.

H(P, Q) = sqrt( 0.5 * integral( (sqrt(p) - sqrt(q))^2 dx ) )
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hellinger_distance(p: np.ndarray, q: np.ndarray, grid: np.ndarray) -> float:
    """One-shot Hellinger between two densities on the same grid."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    g = np.asarray(grid, dtype=float)
    if np.isnan(p).any() or np.isnan(q).any():
        return float("nan")
    sp = np.sqrt(np.maximum(p, 0.0))
    sq = np.sqrt(np.maximum(q, 0.0))
    integrand = (sp - sq) ** 2
    from .normalize import trapezoid
    val = 0.5 * trapezoid(integrand, g)
    return float(np.sqrt(max(val, 0.0)))


def compute_hellinger_series(Pt: pd.DataFrame, Qt: pd.DataFrame, grid: np.ndarray) -> pd.Series:
    """
    Compute the Hellinger distance time series between two density panels for
    a single target. Both Pt and Qt are wide DataFrames indexed by date with
    columns g_i.
    """
    common = Pt.index.intersection(Qt.index)
    Ptc = Pt.loc[common]
    Qtc = Qt.loc[common]
    H = [hellinger_distance(Ptc.loc[dt].to_numpy(dtype=float),
                             Qtc.loc[dt].to_numpy(dtype=float),
                             grid) for dt in common]
    return pd.Series(H, index=common, name="hellinger")
