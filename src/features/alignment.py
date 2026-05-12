"""
Build and align target/feature panels.

Reference: Barrau & Douady (2022) ch. 4 §4.2.

Naming follows the author's review: `targets` for the variables we want to
explain (Y) and `features` for the explanatory variables (X). The user may
keep targets in the feature set or not, both are supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class XYSpec:
    """Specification for splitting a wide features DataFrame into Y and X."""
    targets: Sequence[str]
    features: Optional[Sequence[str]] = None  # if None, use all non-target columns


def build_XY(features_df: pd.DataFrame, spec: XYSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature DataFrame into Y (targets) and X (features)."""
    if not isinstance(features_df.index, pd.DatetimeIndex):
        raise TypeError("features_df must have a DatetimeIndex.")

    cols = list(map(str, features_df.columns))
    targets = [str(t) for t in spec.targets]

    missing_targets = [t for t in targets if t not in cols]
    if missing_targets:
        raise KeyError(f"Missing target tickers in features_df: {missing_targets}")

    if spec.features is None:
        feats = [c for c in cols if c not in set(targets)]
    else:
        feats = [str(f) for f in spec.features]
        missing_feats = [f for f in feats if f not in cols]
        if missing_feats:
            raise KeyError(f"Missing feature tickers in features_df: {missing_feats}")

    Y = features_df.loc[:, targets].copy()
    X = features_df.loc[:, feats].copy()
    return Y, X


def align_XY(Y: pd.DataFrame, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align Y and X on their common index and drop rows where X is fully NaN
    (typically the first `lag_days` rows after a shift).
    Per-pair NaN handling is left to the estimation step.
    """
    common = Y.index.intersection(X.index)
    Yc = Y.loc[common].copy()
    Xc = X.loc[common].copy()

    all_nan = Xc.isna().all(axis=1)
    if all_nan.any():
        Yc = Yc.loc[~all_nan]
        Xc = Xc.loc[~all_nan]
    return Yc, Xc
