"""
Pre-crisis (Q_t) and normal-times (R_t) memory distributions.

Reference: Barrau & Douady (2022) ch. 4 section 4.3.2.

Pandas-native EMA implementation with calendar-uniform decay (every
rebalancing period decays by the same alpha). Q is built as a weighted
EMA of P_{t-lag} with weight = fwd_return^2 when the realized forward
return was negative, else weight = 0. The division num/den at each date
yields the conditional weighted average.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .normalize import normalize_density


def alpha_from_halflife_years(halflife_years: float, periods_per_year: int = 12) -> float:
    """Convert a half-life in years to an EMA alpha at the period frequency."""
    hl = float(halflife_years) * float(periods_per_year)
    return 1.0 - float(np.exp(np.log(0.5) / hl))


def forward_cumulative_return(returns: pd.Series, horizon: int) -> pd.Series:
    """
    Forward cumulative compounded return over `horizon` periods, aligned at
    the start period t (so fwd[t] looks at returns t+1 .. t+horizon).
    """
    r = returns.astype(float)
    log1p = np.log1p(r)
    shifted = log1p.shift(-1)
    roll = shifted.rolling(window=horizon, min_periods=horizon).sum()
    fwd = np.expm1(roll)
    return fwd.shift(-(horizon - 1))


def _normalize_rows_df(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Normalize each row of `df` as a density on `grid`."""
    out = df.copy()
    arr = df.to_numpy(dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i]
        if np.all(np.isnan(row)) or np.nansum(row) <= 0:
            continue
        out.iloc[i] = normalize_density(row, grid)
    return out


def build_memory_distributions(
    Pt: pd.DataFrame,
    grid: np.ndarray,
    target_forward_return: pd.Series,
    forward: int = 3,
    halflife_years: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build pre-crisis Q_t and normal-times R_t distributions for a single target.

    Parameters
    ----------
    Pt : MultiIndex (date, target) DataFrame containing exactly one target.
        Pass via `Pt.xs(tgt, level='target', drop_level=False)`.
    grid : the common density grid.
    target_forward_return : forward `forward`-period return at index t,
        produced by `forward_cumulative_return(Y_at_rebal, horizon=forward)`.
        Used to (a) flag bad periods (fwd < 0), and (b) weight by fwd^2.
    forward : the forward horizon, doubles as the look-ahead-prevention lag.
    halflife_years : EMA half-life expressed in calendar years.
    """
    if not isinstance(Pt.index, pd.MultiIndex):
        raise TypeError("Pt must have a MultiIndex (date, target).")
    if Pt.index.names != ["date", "target"]:
        Pt.index.set_names(["date", "target"], inplace=True)

    target = Pt.index.get_level_values("target")[0]
    Pt_wide = Pt.xs(target, level="target")  # date x grid_columns
    Pt_wide = Pt_wide.sort_index()

    halflife_periods = float(halflife_years) * 12.0  # rebalancing is monthly

    # Normal-times: straight EMA across all dates
    Rt = Pt_wide.ewm(halflife=halflife_periods, adjust=False).mean()

    # Pre-crisis: weighted EMA of P_{t-lag}, weight = fwd^2 when fwd<0 else 0
    # `forward` doubles as the lag (the realized fwd_t covers t+1..t+forward,
    # so we only know it after `forward` periods, hence shift by `forward`).
    fwd = target_forward_return.reindex(Pt_wide.index)
    weights = (fwd ** 2).where(fwd < 0, 0.0).fillna(0.0)

    weights_shift = weights.shift(forward).fillna(0.0)
    Pt_shift = Pt_wide.shift(forward).fillna(0.0)

    weighted_Pt = Pt_shift.mul(weights_shift, axis=0)
    num = weighted_Pt.ewm(halflife=halflife_periods, adjust=False).mean()
    den = weights_shift.ewm(halflife=halflife_periods, adjust=False).mean()

    # Avoid division by zero before any bad period has been observed
    Qt = num.div(den.replace(0.0, np.nan), axis=0)

    Rt = _normalize_rows_df(Rt, grid)
    Qt = _normalize_rows_df(Qt, grid)
    return Qt, Rt
