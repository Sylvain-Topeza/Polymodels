from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .normalize import normalize_density


@dataclass(frozen=True)
class MemoryParams:
    forward_months: int = 3
    halflife_years: float = 10.0
    lag_months: int = 3  # anti look-ahead: update at t using info from t-lag

    def alpha_monthly(self) -> float:
        """
        Convert half-life in years into a monthly EMA alpha.
        half-life years -> half-life months = years * 12
        alpha = 1 - exp(log(0.5)/HL)
        """
        hl_months = float(self.halflife_years) * 12.0
        return 1.0 - float(np.exp(np.log(0.5) / hl_months))


def forward_cumulative_return(monthly_returns: pd.Series, horizon_months: int) -> pd.Series:
    """
    Compute forward horizon cumulative return using compounding across the next `horizon_months`.
    Output is indexed by the start month t (month-end date):
      fwd[t] = prod_{k=1..h} (1 + r_{t+k}) - 1
    Last `horizon_months` values are NaN.
    """
    r = monthly_returns.astype(float)
    # Using log-sum-exp for stability:
    log1p = np.log1p(r)
    # Shift by -1 to start at t+1, then rolling sum horizon, aligned to t+h
    shifted = log1p.shift(-1)
    roll = shifted.rolling(window=horizon_months, min_periods=horizon_months).sum()
    fwd = np.expm1(roll)
    # roll is aligned at t+h, but we want it aligned at t:
    return fwd.shift(-(horizon_months - 1))


def _weighted_ema_update(
    num_prev: np.ndarray,
    den_prev: float,
    x_density: np.ndarray,
    weight: float,
    alpha: float,
) -> Tuple[np.ndarray, float]:
    """
    Update numerator and denominator of a weighted EMA:
      num_t = (1-a)*num_{t-1} + a*(w * x)
      den_t = (1-a)*den_{t-1} + a*w
    """
    num = (1.0 - alpha) * num_prev + alpha * (weight * x_density)
    den = (1.0 - alpha) * den_prev + alpha * weight
    return num, den


def build_memory_distributions(
    Pt: pd.DataFrame,
    grid: np.ndarray,
    market_forward_3m: pd.Series,
    mem_params: MemoryParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
      - Q_t: pre-crisis memory distribution (weighted EMA of past P where forward return < 0)
      - R_t: normal-times distribution (EMA of all P)

    Pt is a wide dataframe:
      index: MultiIndex (date, market)
      columns: g_0 ... g_{G-1} (normalized)
    market_forward_3m is indexed by month-end date and corresponds to ONE market.
    This function should be called per market because Pt contains a single unique market.

    Returns
    -------
    Qt_wide, Rt_wide:
      index: month-end dates (aligned to Pt dates for that market)
      columns: g_i
    """
    if not isinstance(Pt.index, pd.MultiIndex):
        raise TypeError("Pt must have a MultiIndex (date, market).")
    if Pt.index.names != ["date", "market"]:
        Pt.index.set_names(["date", "market"], inplace=True)

    alpha = mem_params.alpha_monthly()
    lag = mem_params.lag_months
    dates = sorted(Pt.index.get_level_values("date").unique())

    G = Pt.shape[1]
    cols = list(Pt.columns)

    # Numerator/denominator states for Q (precrisis)
    q_num = np.zeros(G, dtype=float)
    q_den = 0.0

    # For R (normal-times)
    r_num = np.zeros(G, dtype=float)
    r_den = 0.0

    Qt_rows = []
    Rt_rows = []
    out_dates = []

    # Create fast lookup: date -> P_density vector
    # Pt for a single market will be passed (subset externally), so just date index works
    P_by_date = {dt: Pt.loc[(dt, Pt.index.get_level_values("market")[0]), :].to_numpy(dtype=float).reshape(-1)
                 for dt in dates}

    for i, t in enumerate(dates):
        # Always update normal-times with weight=1 using P_t (known at time t)
        P_t = P_by_date[t]
        r_num, r_den = _weighted_ema_update(r_num, r_den, P_t, weight=1.0, alpha=alpha)

        # Pre-crisis update uses past date = t - lag
        if i >= lag:
            t_past = dates[i - lag]
            fwd_ret = market_forward_3m.get(t_past, np.nan)

            if np.isfinite(fwd_ret) and fwd_ret < 0:
                w = float(fwd_ret ** 2)
                P_past = P_by_date[t_past]
                q_num, q_den = _weighted_ema_update(q_num, q_den, P_past, weight=w, alpha=alpha)

        # Build densities if denominators positive
        if q_den > 0:
            Q_t = q_num / q_den
            Q_t = normalize_density(Q_t, grid)
        else:
            Q_t = np.full(G, np.nan)

        if r_den > 0:
            R_t = r_num / r_den
            R_t = normalize_density(R_t, grid)
        else:
            R_t = np.full(G, np.nan)

        Qt_rows.append(Q_t)
        Rt_rows.append(R_t)
        out_dates.append(t)

    Qt_wide = pd.DataFrame(np.vstack(Qt_rows), index=pd.DatetimeIndex(out_dates), columns=cols)
    Rt_wide = pd.DataFrame(np.vstack(Rt_rows), index=pd.DatetimeIndex(out_dates), columns=cols)
    return Qt_wide, Rt_wide