"""
Configuration constants for the Polymodel pipeline.

Reference: Barrau & Douady (2022), Artificial Intelligence for Financial
Markets, The Polymodel Approach (Springer), chapters 1 to 4.

Defaults follow the book unless explicitly noted. The configuration is a
plain dict so it serializes trivially and is hashable for run-id derivation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def default_config() -> Dict[str, Any]:
    """
    Return a fresh copy of the default configuration.
    Override values in the notebook by mutating the returned dict.
    """
    return {
        # --- Targets and feature metric (book ch. 4 §4.2) ---
        "targets": ["SPX Index"],
        "features": None, # None = all non-targets, else list of feature tickers
        "metric": "rmse",                       # "rmse" or "correlation" (variant)

        # --- Rolling features (book ch. 4 §4.3) ---
        "rolling_window_days": 21,              # 1-month equivalent
        "rolling_min_periods": 18,              # holiday tolerance (18/21)

        # --- LNLM panel estimation (book ch. 3) ---
        "window_days": 1260,                    # 5 years of daily data
        "min_obs_per_model": 252,               # at least 1 valid year
        "lag_days": 21,                         # X_{t-21} -> Y_t, disjoint windows
        "degree_n": 4,                          # Hermite polynomials up to degree 4
        "n_folds": 5,                           # k-fold cross-validation
        "n_mu_points": 100,                     # mu grid size
        "random_state": 42,                     # default seed for reproducibility
        "n_workers": -1,                        # joblib: -1 = all cores, 1 = sequential
        "joblib_verbose": 5,                    # joblib progress verbosity (0 silent, 10+ very detailed)

        # --- KDE on the RMSE distribution (book ch. 4 §4.3.2) ---
        "kde_grid_size": 400,
        "kde_bandwidth": None,                  # None -> Silverman
        "keep_top_quantile": 0.80,              # filter out the 20% worst RMSE
        "min_samples": 10,                      # minimum factors for a valid KDE

        # --- Pre-crisis memory (book ch. 4 §4.3.2) ---
        "forward_months": 3,                    # 3-month forward return horizon
        "memory_halflife_years": 10.0,          # exponential decay half-life

        # --- Cleaning ---
        "clip_abs_max": None,                   # symmetric clip on returns, e.g. 0.20

        # --- Backtest (book ch. 4 §4.4.1) ---
        "leverage_risk_on": 2.0,
        "leverage_risk_off": 0.5,

        # --- Paths ---
        "artifacts_root": "data/artifacts",
        "results_root": "results",
    }


def config_hash(config: Dict[str, Any]) -> str:
    """Stable 10-character hash, used to version artifacts under a single run id."""
    payload = json.dumps(config, indent=2, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]
