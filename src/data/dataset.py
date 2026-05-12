"""
Data layer for the Polymodel pipeline.

Reference: Barrau & Douady (2022), Artificial Intelligence for Financial
Markets, The Polymodel Approach (Springer), ch. 4 §4.2 (Data).

Three pandas-native functions, DataFrame in / DataFrame out:

    format_dataset(...)   single entry point used by the notebook
    load_transform(...)   dictionary -> (price_tickers, rate_tickers) + warnings
    compute_returns(...)  prices -> returns with rates handled separately

Conventions:
- The user supplies DataFrames. We never read files here.
- The file is king: column names and the time index are taken as-is. We warn
  on suspicious shapes but never silently repair them.
- Default ticker type is "price". Rates are opt-in via the dictionary.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def format_dataset(
    prices: Optional[pd.DataFrame] = None,
    returns: Optional[pd.DataFrame] = None,
    dictionary: Optional[pd.DataFrame] = None,
    *,
    symbol_col: str = "SYMBOL",
    nature_col: str = "Nature",
    rate_label: str = "Rate",
    clip_abs_max: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build the standardized returns DataFrame consumed by the rest of the pipeline.

    Exactly one of `prices` or `returns` must be provided (XOR). If `prices` is
    given, returns are computed via `compute_returns`. Either way, infinities
    are dropped early and columns are split into price/rate buckets via
    `load_transform`.

    Parameters
    ----------
    prices, returns : DataFrame indexed by date, columns are tickers.
    dictionary : optional DataFrame with at least columns `symbol_col` and
        `nature_col`. Tickers whose `nature_col` equals `rate_label` are
        treated as rates.
    clip_abs_max : optional symmetric clipping threshold applied to the final
        returns. Same value used as lower and upper bound, so users who have
        not pre-clipped get a sane safety net.

    Returns
    -------
    returns_df : DataFrame of cleaned daily returns.
    price_tickers : list of column names treated as prices.
    rate_tickers : list of column names treated as rates.
    """
    if (prices is None) == (returns is None):
        raise ValueError("Provide exactly one of `prices` or `returns`, not both or neither.")

    df = prices if prices is not None else returns
    _check_index(df, name="prices" if prices is not None else "returns")

    price_tickers, rate_tickers = load_transform(
        dictionary=dictionary,
        columns=list(df.columns),
        symbol_col=symbol_col,
        nature_col=nature_col,
        rate_label=rate_label,
    )

    if prices is not None:
        returns_df = compute_returns(
            prices=prices,
            rate_tickers=rate_tickers,
            clip_abs_max=clip_abs_max,
        )
    else:
        returns_df = returns.replace([np.inf, -np.inf], np.nan)
        returns_df = returns_df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if clip_abs_max is not None:
            returns_df = returns_df.clip(lower=-clip_abs_max, upper=clip_abs_max)

    # Re-sync price/rate lists with whatever survived the cleanup.
    surviving = set(map(str, returns_df.columns))
    price_tickers = [t for t in price_tickers if t in surviving]
    rate_tickers = [t for t in rate_tickers if t in surviving]

    return returns_df, price_tickers, rate_tickers


def load_transform(
    dictionary: Optional[pd.DataFrame],
    columns: List[str],
    *,
    symbol_col: str = "SYMBOL",
    nature_col: str = "Nature",
    rate_label: str = "Rate",
) -> Tuple[List[str], List[str]]:
    """
    Split `columns` into a price list and a rate list using the dictionary.

    If `dictionary` is None, every column defaults to price (with a warning).
    Tickers present in `columns` but absent from the dictionary default to
    price, with a warning. Tickers present in the dictionary but absent from
    `columns` trigger a separate warning.
    """
    if dictionary is None:
        warnings.warn(
            "No dictionary provided: every ticker is treated as a Price series. "
            "If your dataset contains rates, pass a dictionary DataFrame to handle them.",
            UserWarning,
            stacklevel=2,
        )
        return list(columns), []

    if symbol_col not in dictionary.columns or nature_col not in dictionary.columns:
        raise KeyError(
            f"Dictionary must contain columns '{symbol_col}' and '{nature_col}'. "
            f"Got: {list(dictionary.columns)}"
        )

    rate_set = set(
        dictionary.loc[dictionary[nature_col] == rate_label, symbol_col].astype(str)
    )
    known_set = set(dictionary[symbol_col].astype(str))

    price_tickers: List[str] = []
    rate_tickers: List[str] = []
    unknown: List[str] = []
    for col in columns:
        c = str(col)
        if c in rate_set:
            rate_tickers.append(c)
        elif c in known_set:
            price_tickers.append(c)
        else:
            price_tickers.append(c)
            unknown.append(c)

    if unknown:
        warnings.warn(
            f"{len(unknown)} ticker(s) not in dictionary, defaulted to Price: "
            f"{unknown[:5]}{' ...' if len(unknown) > 5 else ''}",
            UserWarning,
            stacklevel=2,
        )
    missing_in_data = sorted(known_set - set(map(str, columns)))
    if missing_in_data:
        warnings.warn(
            f"{len(missing_in_data)} dictionary ticker(s) not present in data: "
            f"{missing_in_data[:5]}{' ...' if len(missing_in_data) > 5 else ''}",
            UserWarning,
            stacklevel=2,
        )

    return price_tickers, rate_tickers


def compute_returns(
    prices: pd.DataFrame,
    rate_tickers: Optional[List[str]] = None,
    clip_abs_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convert a prices DataFrame into a daily returns DataFrame.

    For Price-type tickers, the standard percentage return:

        r_t = (p_t - p_{t-1}) / |p_{t-1}|

    The absolute value in the denominator handles negative prices (e.g. oil
    futures), per the author's data notebook.

    For Rate-type tickers, we work in fractional units:

        r_t = (p_t - p_{t-1}) / 100

    Cleaning steps, in this order:
      1. Replace 0 with NaN before dividing (avoids spurious infinities).
      2. Replace remaining +/-inf with NaN.
      3. Drop fully-empty rows and columns.
      4. Optional symmetric clipping at +/- clip_abs_max.
    """
    rate_set = set(rate_tickers or [])
    price_cols = [c for c in prices.columns if c not in rate_set]
    rate_cols = [c for c in prices.columns if c in rate_set]

    p = prices.replace(0, np.nan)

    price_returns = (
        p[price_cols]
        .sub(p[price_cols].shift(1))
        .div(p[price_cols].shift(1).abs())
    )
    rate_returns = p[rate_cols].diff() / 100.0

    out = pd.concat([price_returns, rate_returns], axis=1)[list(prices.columns)]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(how="all", axis=0).dropna(how="all", axis=1)

    if clip_abs_max is not None:
        out = out.clip(lower=-clip_abs_max, upper=clip_abs_max)

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_index(df: pd.DataFrame, name: str) -> None:
    """Warn (do not fix) if the index does not look like a sorted DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        warnings.warn(
            f"{name}.index is expected to be a DatetimeIndex, got {type(df.index).__name__}. "
            "Convert via `df.index = pd.to_datetime(df.index)` before calling.",
            UserWarning,
            stacklevel=3,
        )
        return
    if df.index.has_duplicates:
        warnings.warn(
            f"{name} index contains duplicate dates. Deduplicate before calling.",
            UserWarning,
            stacklevel=3,
        )
    if not df.index.is_monotonic_increasing:
        warnings.warn(
            f"{name} index is expected to be sorted ascending. Sort before calling.",
            UserWarning,
            stacklevel=3,
        )
