from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .normalize import normalize_density


@dataclass(frozen=True)
class KDEParams:
    grid_size: int = 400
    bandwidth: Optional[float] = None  # if None, use Silverman
    keep_top_quantile: float = 0.80    # keep best 80% tail by RMSE (<= q)
    min_samples: int = 10              # minimum RMSE values needed to build a KDE


def build_global_grid_from_rmse(rmse_long: pd.DataFrame, grid_size: int = 400) -> np.ndarray:
    """
    Build a global grid for KDE based on RMSE values.
    Grid starts at 0 and ends slightly above max RMSE.

    rmse_long expected columns: ['date', 'market', 'factor', 'rmse', ...]
    """
    vals = rmse_long["rmse"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, grid_size)

    mx = float(np.nanmax(vals))
    if mx <= 0:
        mx = 1.0
    return np.linspace(0.0, 1.05 * mx, grid_size)


def _silverman_bandwidth(samples: np.ndarray) -> float:
    """
    Silverman's rule of thumb bandwidth for 1D KDE.
    """
    x = samples[np.isfinite(samples)]
    n = x.size
    if n <= 1:
        return 1.0
    std = float(np.std(x, ddof=1)) if n > 1 else float(np.std(x))
    if std <= 1e-12:
        # fallback if all values identical
        std = max(1e-6, float(np.mean(np.abs(x))) * 0.1)
    return 1.06 * std * (n ** (-1.0 / 5.0))


def gaussian_kde_1d(samples: np.ndarray, grid: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
    """
    Simple Gaussian KDE on a fixed grid (1D).

    density(x) = 1/(n*h*sqrt(2pi)) * sum exp(-0.5*((x - xi)/h)^2)
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    g = np.asarray(grid, dtype=float)

    n = x.size
    if n == 0:
        return np.full_like(g, np.nan)

    h = float(bandwidth) if bandwidth is not None else _silverman_bandwidth(x)
    h = max(h, 1e-8)

    # Vectorized computation: (G, N) matrix
    z = (g[:, None] - x[None, :]) / h
    kern = np.exp(-0.5 * z * z)
    dens = kern.sum(axis=1) / (n * h * np.sqrt(2.0 * np.pi))
    return dens


def _filter_tail(values: np.ndarray, keep_top_quantile: float) -> np.ndarray:
    """
    Keep RMSE values >= quantile(keep_top_quantile).
    E.g. keep_top_quantile=0.80 -> keep worst 20% of RMSE values.
    Thus, to keep best 80%, we keep values <= quantile(0.80).
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return v
    q = float(np.quantile(v, keep_top_quantile))
    return v[v <= q]


def build_Pt_distributions(
    rmse_long: pd.DataFrame,
    grid: np.ndarray,
    kde_params: KDEParams,
) -> pd.DataFrame:
    """
    Build current-time RMSE distributions P_t for each (date, market).

    Returns a wide dataframe:
      index: MultiIndex (date, market)
      columns: g_0 ... g_{G-1}
      values: normalized density over grid
    """
    required = {"date", "market", "rmse"}
    missing = required - set(rmse_long.columns)
    if missing:
        raise KeyError(f"rmse_long missing columns: {missing}")

    df = rmse_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["market"] = df["market"].astype(str)

    # Group and compute
    rows = []
    index_tuples = []

    for (dt, mkt), grp in df.groupby(["date", "market"], sort=True):
        vals = grp["rmse"].to_numpy(dtype=float)
        vals = _filter_tail(vals, kde_params.keep_top_quantile)
        if vals.size < kde_params.min_samples:
            continue

        dens = gaussian_kde_1d(vals, grid, bandwidth=kde_params.bandwidth)
        dens = normalize_density(dens, grid)

        if np.isnan(dens).any():
            continue

        rows.append(dens)
        index_tuples.append((dt, mkt))

    if not rows:
        cols = [f"g_{i}" for i in range(len(grid))]
        return pd.DataFrame(columns=cols, index=pd.MultiIndex.from_tuples([], names=["date", "market"]))

    mat = np.vstack(rows)
    cols = [f"g_{i}" for i in range(mat.shape[1])]
    out = pd.DataFrame(mat, columns=cols, index=pd.MultiIndex.from_tuples(index_tuples, names=["date", "market"]))
    out = out.sort_index()
    return out