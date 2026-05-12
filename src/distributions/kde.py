"""
KDE on the RMSE distribution.

Reference: Barrau & Douady (2022) ch. 4 §4.3.2.

For each (date, target), compute a Gaussian KDE of the RMSE values across
features, after filtering out the worst quantile of RMSEs (book default:
keep best 80%). Distributions are normalized to integrate to 1 on the grid.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .normalize import normalize_density


def build_global_grid_from_rmse(rmse_long: pd.DataFrame, grid_size: int = 400) -> np.ndarray:
    """A common grid covering the full range of observed RMSEs, with a small margin."""
    vals = rmse_long["rmse"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, grid_size)
    mx = float(np.nanmax(vals))
    if mx <= 0:
        mx = 1.0
    return np.linspace(0.0, 1.05 * mx, grid_size)


def _silverman_bandwidth(samples: np.ndarray) -> float:
    x = samples[np.isfinite(samples)]
    n = x.size
    if n <= 1:
        return 1.0
    std = float(np.std(x, ddof=1)) if n > 1 else float(np.std(x))
    if std <= 1e-12:
        std = max(1e-6, float(np.mean(np.abs(x))) * 0.1)
    return 1.06 * std * (n ** (-1.0 / 5.0))


def gaussian_kde_1d(samples: np.ndarray, grid: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
    """1D Gaussian KDE evaluated on `grid`."""
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    g = np.asarray(grid, dtype=float)
    if x.size == 0:
        return np.full_like(g, np.nan)
    h = float(bandwidth) if bandwidth is not None else _silverman_bandwidth(x)
    h = max(h, 1e-8)
    z = (g[:, None] - x[None, :]) / h
    kern = np.exp(-0.5 * z * z)
    return kern.sum(axis=1) / (x.size * h * np.sqrt(2.0 * np.pi))


def _filter_tail(values: np.ndarray, keep_top_quantile: float) -> np.ndarray:
    """
    Keep RMSE values <= quantile(keep_top_quantile).
    Convention: keep_top_quantile=0.80 keeps the best 80% of fits (lowest RMSE).
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return v
    q = float(np.quantile(v, keep_top_quantile))
    return v[v <= q]


def build_Pt_distributions(
    rmse_long: pd.DataFrame,
    grid: np.ndarray,
    bandwidth: Optional[float] = None,
    keep_top_quantile: float = 0.80,
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    Build per-(date, target) RMSE densities.

    Returns a wide DataFrame with MultiIndex (date, target) and columns
    g_0 ... g_{G-1} (densities normalized on `grid`).
    """
    required = {"date", "target", "rmse"}
    missing = required - set(rmse_long.columns)
    if missing:
        raise KeyError(f"rmse_long missing columns: {missing}")

    df = rmse_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["target"] = df["target"].astype(str)

    rows = []
    index_tuples = []
    for (dt, tgt), grp in df.groupby(["date", "target"], sort=True):
        vals = grp["rmse"].to_numpy(dtype=float)
        vals = _filter_tail(vals, keep_top_quantile)
        if vals.size < min_samples:
            continue
        dens = gaussian_kde_1d(vals, grid, bandwidth=bandwidth)
        dens = normalize_density(dens, grid)
        if np.isnan(dens).any():
            continue
        rows.append(dens)
        index_tuples.append((dt, tgt))

    if not rows:
        cols = [f"g_{i}" for i in range(len(grid))]
        return pd.DataFrame(columns=cols, index=pd.MultiIndex.from_tuples([], names=["date", "target"]))

    mat = np.vstack(rows)
    cols = [f"g_{i}" for i in range(mat.shape[1])]
    out = pd.DataFrame(
        mat,
        columns=cols,
        index=pd.MultiIndex.from_tuples(index_tuples, names=["date", "target"]),
    )
    return out.sort_index()
