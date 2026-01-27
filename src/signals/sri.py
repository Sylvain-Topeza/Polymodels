from __future__ import annotations

import pandas as pd


def sri_from_hellinger(H: pd.Series) -> pd.Series:
    """
    Systemic Risk Indicator (SRI) rule from the book:
      SRI_t = 1 if H_t < H_{t-1}  (distance decreasing)
            = 0 otherwise

    Returns a series aligned with H (first value is NaN -> 0 by convention).
    """
    dH = H.diff()
    sri = (dH < 0).astype(float)
    sri.iloc[0] = 0.0
    sri.name = "sri"
    return sri


def delta_hellinger(H: pd.Series) -> pd.Series:
    dH = H.diff()
    dH.name = "delta_hellinger"
    return dH