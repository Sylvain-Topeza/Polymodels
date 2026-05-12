"""
Systemic Risk Indicator (SRI).

Reference: Barrau & Douady (2022) ch. 4 section 4.4.

Two rules are provided:

1. State-based : SRI_t = 1 if H(P_t, Q_t) < H(P_t, R_t). The current RMSE
   distribution P is closer to the pre-crisis memory Q than to the normal-times
   memory R, signaling a pre-crisis regime. This rule uses both memory distributions
   and produces persistent block-shaped activations.

2. Derivative-based : SRI_t = 1 iff H_t < H_{t-1}. A decreasing Hellinger
   distance between the current RMSE distribution and the pre-crisis distribution
   flags that the current state is starting to resemble pre-crisis behavior.
"""

from __future__ import annotations

import pandas as pd


def sri_from_state_comparison(H_PQ: pd.Series, H_PR: pd.Series) -> pd.Series:
    """SRI=1 if current state closer to pre-crisis Q than to normal-times R."""
    common = H_PQ.index.intersection(H_PR.index)
    pq = H_PQ.loc[common]
    pr = H_PR.loc[common]
    sri = (pq < pr).astype(float)
    # NaN in either series -> SRI = 0 (no signal yet)
    mask = pq.isna() | pr.isna()
    sri.loc[mask] = 0.0
    sri.name = "sri"
    return sri


def sri_from_hellinger(H: pd.Series) -> pd.Series:
    """Build a binary SRI Series from a Hellinger distance Series."""
    dH = H.diff()
    sri = (dH < 0).astype(float)
    sri.iloc[0] = 0.0
    sri.name = "sri"
    return sri
