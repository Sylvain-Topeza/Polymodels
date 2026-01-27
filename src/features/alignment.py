from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class XYSpec:
    """
    Specification for building X/Y monthly matrices.
    """
    markets: Sequence[str]
    factors: Optional[Sequence[str]] = None  # if None => all columns excluding markets
    lag_months: int = 1  # X_{t-lag} predicts Y_t


def build_XY_monthly(features_monthly: pd.DataFrame, spec: XYSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Y (markets) and X (factors) from a monthly feature dataframe.

    Parameters
    ----------
    features_monthly:
        DataFrame indexed by month-end dates, columns=tickers
    spec:
        markets: list of market tickers (targets)
        factors: list of factor tickers (predictors). If None, use all non-market tickers.

    Returns
    -------
    Y, X:
        Y: DataFrame (T x M) for markets
        X: DataFrame (T x K) for factors
    """
    if not isinstance(features_monthly.index, pd.DatetimeIndex):
        raise TypeError("features_monthly must have a DatetimeIndex.")

    cols = list(map(str, features_monthly.columns))
    markets = [str(m) for m in spec.markets]

    missing_markets = [m for m in markets if m not in cols]
    if missing_markets:
        raise KeyError(f"Missing market tickers in features_monthly: {missing_markets}")

    if spec.factors is None:
        factors = [c for c in cols if c not in set(markets)]
    else:
        factors = [str(f) for f in spec.factors]
        missing_factors = [f for f in factors if f not in cols]
        if missing_factors:
            raise KeyError(f"Missing factor tickers in features_monthly: {missing_factors}")

    Y = features_monthly.loc[:, markets].copy()
    X = features_monthly.loc[:, factors].copy()
    return Y, X


def apply_monthly_lag(X: pd.DataFrame, lag_months: int) -> pd.DataFrame:
    """
    Shift X by lag_months so that X_lagged[t] corresponds to original X[t-lag].

    For the book's setup (lag=1 month): X_{t-1} -> Y_t
    """
    if lag_months < 0:
        raise ValueError("lag_months must be >= 0")
    if lag_months == 0:
        return X.copy()
    return X.shift(lag_months)


def align_XY_after_lag(Y: pd.DataFrame, X_lagged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align Y and X_lagged on common monthly index and drop rows where X_lagged is fully NaN.
    (We do NOT drop rows for partial NaNs here; estimation will handle per factor.)
    """
    common = Y.index.intersection(X_lagged.index)
    Yc = Y.loc[common].copy()
    Xc = X_lagged.loc[common].copy()

    # Remove rows where all predictors are NaN (common at the beginning due to lag)
    all_nan = Xc.isna().all(axis=1)
    if all_nan.any():
        Yc = Yc.loc[~all_nan]
        Xc = Xc.loc[~all_nan]

    return Yc, Xc


def month_end_validate(df: pd.DataFrame) -> None:
    """
    Lightweight check: index is month-end-ish and monotonic increasing.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("Monthly dataframe index must be sorted ascending.")
    # No strict assertion on exact month-end because 'ME' resample can yield calendar month end.