from __future__ import annotations

from typing import Dict, Literal, Optional
import numpy as np
import pandas as pd

TickerType = Literal["price", "rate"]


def compounded_rolling_return(daily_returns: pd.Series, window: int) -> pd.Series:
    """
    Rolling compounded return over a window:
        prod(1+r) - 1
    Implemented as exp(sum(log1p(r))) - 1 to avoid overflow and because pandas has no rolling product.
    """
    r = daily_returns.astype(float)
    # log1p is stable for small returns
    log1p = np.log1p(r)
    roll = log1p.rolling(window=window, min_periods=window).sum()
    return np.expm1(roll)


def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling sum for additive series (e.g. rate differences).
    """
    x = series.astype(float)
    return x.rolling(window=window, min_periods=window).sum()


def compute_21d_compounded_features(
    returns_df: pd.DataFrame,
    ticker_type_map: Dict[str, TickerType],
    window: int = 21,
    rate_method: Literal["sum"] = "sum",
) -> pd.DataFrame:
    """
    Build "monthly returns available at daily frequency" as described in Book ch.4:
    - price series: 21-day compounded return using rolling product
    - rate series: by convention, rolling sum of daily differences (additive)
    """
    out = pd.DataFrame(index=returns_df.index)

    for col in returns_df.columns:
        ttype = ticker_type_map.get(str(col), "price")
        s = returns_df[col]
        if ttype == "rate":
            if rate_method != "sum":
                raise ValueError(f"Unsupported rate_method: {rate_method}")
            out[col] = rolling_sum(s, window=window)
        else:
            out[col] = compounded_rolling_return(s, window=window)

    return out