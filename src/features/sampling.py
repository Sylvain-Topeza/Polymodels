from __future__ import annotations

import pandas as pd


def sample_month_ends(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Sample a daily dataframe at month-end dates (calendar month end).
    Uses pandas resample('ME').last().
    """
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        raise TypeError("df_daily index must be a DatetimeIndex")
    return df_daily.resample("ME").last()


def month_end_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Convenience: month-end index derived from a daily DatetimeIndex.
    """
    s = pd.Series(1, index=index)
    return s.resample("ME").last().index
