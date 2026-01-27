from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .lnlm_fit import fit_lnlm_single_factor, LnlmFitResult


@dataclass(frozen=True)
class EstimationParams:
    """
    Parameters for monthly LNLM panel estimation.
    """
    window_months: int = 60
    min_window_months: int = 30
    degree_n: int = 4
    n_folds: int = 5
    n_mu_points: int = 100
    random_state: Optional[int] = None

def _select_window_indices(index: pd.DatetimeIndex, end_pos: int, window_months: int) -> slice:
    """
    Return slice selecting the last `window_months` observations ending at end_pos (inclusive).
    """
    start_pos = max(0, end_pos - window_months + 1)
    return slice(start_pos, end_pos + 1)


def run_lnlm_panel_estimation(
    Y_monthly: pd.DataFrame,
    X_lagged_monthly: pd.DataFrame,
    params: EstimationParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run LNLM estimation panel, monthly frequency.

    For each month-end t:
      - use a rolling window of (up to) `window_months` observations ending at t
      - if available obs < min_window_months => skip
      - fit one LNLM per (market, factor) using x=X_lagged[:, factor] and y=Y[:, market]
      - compute in-sample rmse and r2
      - store mu and coefficients for refit/audit

    Returns
    -------
    rmse_long, params_long
    """
    if not isinstance(Y_monthly.index, pd.DatetimeIndex) or not isinstance(X_lagged_monthly.index, pd.DatetimeIndex):
        raise TypeError("Y_monthly and X_lagged_monthly must have DatetimeIndex.")

    # Align on common index
    common = Y_monthly.index.intersection(X_lagged_monthly.index)
    Y = Y_monthly.loc[common].copy()
    X = X_lagged_monthly.loc[common].copy()

    dates = list(common)
    markets = list(Y.columns)
    factors = list(X.columns)

    rmse_rows: List[dict] = []
    param_rows: List[dict] = []

    for i, t in enumerate(dates):
        # Determine how many observations are available up to i (inclusive)
        n_avail = i + 1
        if n_avail < params.min_window_months:
            continue

        win_len = min(params.window_months, n_avail)
        sl = _select_window_indices(common, i, win_len)

        Y_win = Y.iloc[sl]
        X_win = X.iloc[sl]

        # For each market/factor pair, fit on the overlapping finite points
        for m in markets:
            y_series = Y_win[m].astype(float)

            for f in factors:
                x_series = X_win[f].astype(float)

                y_arr = y_series.to_numpy(dtype=float)
                x_arr = x_series.to_numpy(dtype=float)
                mask = np.isfinite(y_arr) & np.isfinite(x_arr)

                n_obs = int(mask.sum())
                if n_obs < params.min_obs_per_model:
                    continue

                x = x_arr[mask]
                y = y_arr[mask]

                try:
                    fit: LnlmFitResult = fit_lnlm_single_factor(
                        x=x,
                        y=y,
                        degree_n=params.degree_n,
                        n_folds=params.n_folds,
                        n_mu_points=params.n_mu_points,
                        random_state=params.random_state,
                    )
                except Exception:
                    # If a single fit fails (rare), skip it rather than crashing the full panel
                    continue

                rmse_rows.append(
                    {
                        "date": t,
                        "market": str(m),
                        "factor": str(f),
                        "rmse": fit.rmse,
                        "r2": fit.r2,
                        "n_obs": fit.n_obs,
                    }
                )

                # Expand nonlinear coeffs into separate columns for Parquet friendliness
                row = {
                    "date": t,
                    "market": str(m),
                    "factor": str(f),
                    "mu": fit.mu,
                    "y_mean": fit.y_mean,
                    "linear_coef": fit.linear_coef,
                    "degree_n": fit.degree_n,
                    "n_folds": fit.n_folds,
                    "n_obs": fit.n_obs,
                }
                for j, c in enumerate(fit.nonlinear_coef):
                    row[f"nonlinear_coef_{j}"] = float(c)
                param_rows.append(row)

    rmse_long = pd.DataFrame(rmse_rows)
    params_long = pd.DataFrame(param_rows)

    if not rmse_long.empty:
        rmse_long["date"] = pd.to_datetime(rmse_long["date"])
        rmse_long = rmse_long.sort_values(["date", "market", "factor"]).reset_index(drop=True)

    if not params_long.empty:
        params_long["date"] = pd.to_datetime(params_long["date"])
        params_long = params_long.sort_values(["date", "market", "factor"]).reset_index(drop=True)

    return rmse_long, params_long