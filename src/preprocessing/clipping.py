from __future__ import annotations

from typing import Dict, Literal, Optional
import numpy as np
import pandas as pd

TickerType = Literal["price", "rate"]
ClippingMode = Literal["clip", "nan"]


def clip_series(values: pd.Series, threshold: float, mode: ClippingMode) -> pd.Series:
    """
    Clip a series symmetrically to +/- threshold.
    If mode == "nan", outliers are replaced with NaN.
    """
    if threshold <= 0:
        return values

    if mode == "clip":
        return values.clip(lower=-threshold, upper=threshold)

    if mode == "nan":
        out = values.copy()
        mask = out.abs() > threshold
        out[mask] = np.nan
        return out

    raise ValueError(f"Unknown clipping mode: {mode}")


def clip_returns_by_type(
    returns_df: pd.DataFrame,
    ticker_type_map: Dict[str, TickerType],
    asset_threshold: Optional[float],
    rate_threshold: Optional[float],
    mode: ClippingMode = "clip",
) -> pd.DataFrame:
    """
    Apply different symmetric clipping thresholds depending on ticker type.

    - price tickers: asset_threshold (e.g. 0.20 for +/-20%)
    - rate tickers: rate_threshold (in the units of the rate differences, e.g. 0.01 = 100bps)

    If a ticker is missing from ticker_type_map, it is treated as "price" by default.
    """
    out = returns_df.copy()

    for col in out.columns:
        ttype = ticker_type_map.get(str(col), "price")
        if ttype == "rate":
            if rate_threshold is None:
                continue
            out[col] = clip_series(out[col], rate_threshold, mode)
        else:
            if asset_threshold is None:
                continue
            out[col] = clip_series(out[col], asset_threshold, mode)

    return out


def basic_nan_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleanup:
    - Sort index
    - Drop duplicated timestamps keeping last
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out