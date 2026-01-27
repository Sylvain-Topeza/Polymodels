from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def hellinger_distance(p: np.ndarray, q: np.ndarray, grid: np.ndarray) -> float:
    """
    Discrete Hellinger distance between two densities on the same grid.

    H(P,Q) = sqrt( 0.5 * ∫ (sqrt(p) - sqrt(q))^2 dx )
           = sqrt( 1 - ∫ sqrt(p*q) dx )
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    g = np.asarray(grid, dtype=float)

    if np.isnan(p).any() or np.isnan(q).any():
        return float("nan")

    sp = np.sqrt(np.maximum(p, 0.0))
    sq = np.sqrt(np.maximum(q, 0.0))
    integrand = (sp - sq) ** 2
    val = 0.5 * float(np.trapz(integrand, g))
    return float(np.sqrt(max(val, 0.0)))


def compute_hellinger_series(
    Pt_market: pd.DataFrame,
    Qt_market: pd.DataFrame,
    grid: np.ndarray,
) -> pd.Series:
    """
    Compute Hellinger distance time series for one market.
    Pt_market and Qt_market are wide dfs indexed by date, columns g_i.
    """
    common = Pt_market.index.intersection(Qt_market.index)
    Pt = Pt_market.loc[common]
    Qt = Qt_market.loc[common]

    H = []
    for dt in common:
        p = Pt.loc[dt].to_numpy(dtype=float)
        q = Qt.loc[dt].to_numpy(dtype=float)
        H.append(hellinger_distance(p, q, grid))

    return pd.Series(H, index=common, name="hellinger")