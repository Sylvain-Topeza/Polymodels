"""
Rolling features for the Polymodel pipeline.

Reference: Barrau & Douady (2022) ch. 4 §4.3.

For Price-type tickers we build the compounded return over `window` days at
daily frequency:

    r_w = prod_{k=0..w-1}(1 + r_{t-k}) - 1

For Rate-type tickers we build the rolling sum over `window` days:

    r_w = sum_{k=0..w-1} r_{t-k}

`min_periods` allows a small tolerance for holidays (book hint: 18/21).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def rolling_compounded_return(returns: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    """Compounded return over `window` days, applied column-wise."""
    log1p = np.log1p(returns)
    rolled = log1p.rolling(window=window, min_periods=min_periods).sum()
    return np.expm1(rolled)


def rolling_sum(returns: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    """Plain rolling sum over `window` days, for additive series like rate diffs."""
    return returns.rolling(window=window, min_periods=min_periods).sum()


def build_rolling_features(
    returns: pd.DataFrame,
    price_tickers: List[str],
    rate_tickers: List[str],
    window: int = 21,
    min_periods: int = 18,
) -> pd.DataFrame:
    """
    Build the rolling-window features at daily frequency.

    Parameters
    ----------
    returns : DataFrame of daily returns (output of `format_dataset`).
    price_tickers, rate_tickers : column lists produced by `format_dataset`.
    window : rolling window length in days (book default: 21).
    min_periods : minimum non-NaN observations to emit a value.

    Returns
    -------
    DataFrame with the same shape as `returns`, columns reordered as in the
    input. Tickers absent from both lists are silently dropped (defensive).
    """
    price_cols = [c for c in price_tickers if c in returns.columns]
    rate_cols = [c for c in rate_tickers if c in returns.columns]

    price_part = rolling_compounded_return(returns[price_cols], window, min_periods)
    rate_part = rolling_sum(returns[rate_cols], window, min_periods)

    out = pd.concat([price_part, rate_part], axis=1)
    # Restore the input column order, keeping only the kept columns
    keep = [c for c in returns.columns if c in price_cols or c in rate_cols]
    return out[keep]
