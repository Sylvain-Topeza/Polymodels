"""
LNLM panel estimation.

Reference: Barrau & Douady (2022) ch. 4 section 4.3.

For each rebalancing date and each (target, feature) pair, fit one LNLM on a
rolling window of `window_days` daily observations ending at the rebalancing
date. Pairs with fewer than `min_obs_per_model` valid joint observations in
the window are skipped.

Parallelization via joblib.Parallel has each worker handle one feature across
all dates and targets. Joblib's default loky backend uses cloudpickle and works
in Jupyter notebooks on Windows out of the box, unlike raw multiprocessing.Pool.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from joblib import Parallel, delayed

from .lnlm_fit import LnlmFitResult, fit_lnlm_single_factor
from ..features.alignment import align_XY


def _process_feature_for_panel(
    feature_name: str,
    x_series: pd.Series,
    Y: pd.DataFrame,
    rebalancing_dates: pd.DatetimeIndex,
    window_days: int,
    min_obs: int,
    degree_n: int,
    n_folds: int,
    n_mu_points: int,
    random_state: int,
) -> Tuple[List[dict], List[dict]]:
    """Worker: fit LNLM for one feature across all (date, target). Module-level for joblib."""
    common = x_series.index
    targets = list(Y.columns)
    rmse_rows: List[dict] = []
    param_rows: List[dict] = []

    for t in rebalancing_dates:
        if t not in common:
            continue
        end_pos = common.get_loc(t)
        start_pos = max(0, end_pos - window_days + 1)
        x_win = x_series.iloc[start_pos : end_pos + 1]
        Y_win = Y.iloc[start_pos : end_pos + 1]

        for tgt in targets:
            pair = pd.concat([x_win, Y_win[tgt]], axis=1, keys=["x", "y"]).dropna()
            if len(pair) < min_obs:
                continue
            try:
                fit: LnlmFitResult = fit_lnlm_single_factor(
                    x=pair["x"].to_numpy(),
                    y=pair["y"].to_numpy(),
                    degree_n=degree_n,
                    n_folds=n_folds,
                    n_mu_points=n_mu_points,
                    random_state=random_state,
                )
            except Exception as e:
                print(f"Fit failed for {tgt} / {feature_name} at {t}: {type(e).__name__} - {e}")
                continue

            rmse_rows.append({
                "date": t,
                "target": str(tgt),
                "feature": str(feature_name),
                "rmse": fit.rmse,
                "r2": fit.r2,
                "n_obs": fit.n_obs,
            })
            row = {
                "date": t,
                "target": str(tgt),
                "feature": str(feature_name),
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

    return rmse_rows, param_rows


def fit_lnlm_panel(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    rebalancing_dates: pd.DatetimeIndex,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit one LNLM per (target, feature, rebalancing_date) triple.

    Parameters
    ----------
    Y : DataFrame of target time series, indexed by daily dates.
    X : DataFrame of feature time series, indexed by daily dates, already
        lagged by `config["lag_days"]` if applicable.
    rebalancing_dates : the dates at which to refit the panel.
    config : configuration dict produced by `default_config()`. Read keys:
        window_days, min_obs_per_model, degree_n, n_folds, n_mu_points,
        random_state, n_workers, joblib_verbose.

    Returns
    -------
    rmse_long : long DataFrame with columns date, target, feature, rmse, r2, n_obs.
    params_long : long DataFrame with fit parameters (mu, coefs, etc.).
    """
    Y, X = align_XY(Y, X)
    features = list(X.columns)

    window_days = config["window_days"]
    min_obs = config["min_obs_per_model"]
    degree_n = config["degree_n"]
    n_folds = config["n_folds"]
    n_mu_points = config["n_mu_points"]
    random_state = config["random_state"]
    n_workers = config.get("n_workers", -1)
    joblib_verbose = config.get("joblib_verbose", 5)

    if n_workers in (1, None):
        results = [
            _process_feature_for_panel(
                feat, X[feat], Y, rebalancing_dates,
                window_days, min_obs, degree_n, n_folds, n_mu_points, random_state,
            )
            for feat in features
        ]
    else:
        results = Parallel(n_jobs=n_workers, verbose=joblib_verbose)(
            delayed(_process_feature_for_panel)(
                feat, X[feat], Y, rebalancing_dates,
                window_days, min_obs, degree_n, n_folds, n_mu_points, random_state,
            )
            for feat in features
        )

    rmse_rows = [r for rmse, _ in results for r in rmse]
    param_rows = [r for _, params in results for r in params]

    rmse_long = pd.DataFrame(rmse_rows)
    params_long = pd.DataFrame(param_rows)
    if not rmse_long.empty:
        rmse_long["date"] = pd.to_datetime(rmse_long["date"])
        rmse_long = rmse_long.sort_values(["date", "target", "feature"]).reset_index(drop=True)
    if not params_long.empty:
        params_long["date"] = pd.to_datetime(params_long["date"])
        params_long = params_long.sort_values(["date", "target", "feature"]).reset_index(drop=True)
    return rmse_long, params_long
