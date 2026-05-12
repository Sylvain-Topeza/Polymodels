"""
Monthly-rebalanced backtest.

Reference: Barrau & Douady (2022) ch. 4 §4.4.1.

Convention: SRI observed at month p determines the leverage applied during
month p+1. Leverage is held constant within each month and applied daily.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _extract_sri_wide(sri_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce SRI input into wide format: index=date, columns=targets."""
    if isinstance(sri_df.index, pd.MultiIndex):
        if sri_df.index.names != ["date", "target"]:
            sri_df = sri_df.copy()
            sri_df.index.set_names(["date", "target"], inplace=True)
        if "sri" not in sri_df.columns and sri_df.shape[1] == 1:
            sri_df = sri_df.rename(columns={sri_df.columns[0]: "sri"})
        wide = sri_df.reset_index().pivot(index="date", columns="target", values="sri")
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()
    wide = sri_df.copy()
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def build_monthly_positions_from_sri(
    sri_monthly: pd.Series,
    leverage_risk_on: float = 2.0,
    leverage_risk_off: float = 0.5,
    risk_off_when_sri_is_one: bool = True,
    default_leverage: float = 1.0,
) -> pd.Series:
    """SRI in {0,1} -> leverage at month p+1, indexed by PeriodIndex('M')."""
    if not isinstance(sri_monthly.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("sri_monthly index must be DatetimeIndex or PeriodIndex.")
    if isinstance(sri_monthly.index, pd.DatetimeIndex):
        sri_by_month = sri_monthly.copy()
        sri_by_month.index = sri_by_month.index.to_period("M")
    else:
        sri_by_month = sri_monthly.copy()

    s = sri_by_month.astype(float)
    if risk_off_when_sri_is_one:
        lev = s.map(lambda v: leverage_risk_off if v >= 0.5 else leverage_risk_on)
    else:
        lev = s.map(lambda v: leverage_risk_on if v >= 0.5 else leverage_risk_off)

    positions = lev.shift(1).fillna(default_leverage)
    positions.name = "leverage"
    return positions


def expand_monthly_positions_to_daily(
    daily_index: pd.DatetimeIndex,
    positions_by_month: pd.Series,
    default_leverage: float = 1.0,
) -> pd.Series:
    """Broadcast monthly leverage to a daily index."""
    if not isinstance(daily_index, pd.DatetimeIndex):
        raise TypeError("daily_index must be a DatetimeIndex.")
    if not isinstance(positions_by_month.index, pd.PeriodIndex):
        raise TypeError("positions_by_month must have a PeriodIndex.")
    day_month = daily_index.to_period("M")
    all_months = pd.PeriodIndex(day_month.unique(), freq="M").sort_values()
    pos_full = positions_by_month.reindex(all_months).fillna(default_leverage)
    return pd.Series(pos_full.loc[day_month].to_numpy(), index=daily_index, name="leverage")


def run_monthly_backtest(
    daily_returns: pd.DataFrame,
    sri_df: pd.DataFrame,
    targets: Sequence[str],
    leverage_risk_on: float = 2.0,
    leverage_risk_off: float = 0.5,
    risk_off_when_sri_is_one: bool = True,
    default_leverage: float = 1.0,
    fill_missing_with_zero: bool = True,
) -> Dict[str, Any]:
    """Run the multi-target monthly-rebalanced backtest."""
    rets = _ensure_datetime_index(daily_returns)
    missing = [t for t in targets if t not in rets.columns]
    if missing:
        raise KeyError(f"Missing target columns in daily returns: {missing}")

    sri_wide = _extract_sri_wide(sri_df)
    missing_sri = [t for t in targets if t not in sri_wide.columns]
    if missing_sri:
        raise KeyError(f"Missing target columns in SRI: {missing_sri}")

    daily_leverage = pd.DataFrame(index=rets.index, columns=targets, dtype=float)
    strat_daily = pd.DataFrame(index=rets.index, columns=targets, dtype=float)
    equity = pd.DataFrame(index=rets.index, columns=targets, dtype=float)
    bh_equity = pd.DataFrame(index=rets.index, columns=targets, dtype=float)
    monthly_positions: Dict[str, pd.Series] = {}

    for tgt in targets:
        r = rets[tgt]
        sri_t = sri_wide[tgt].dropna()
        if not isinstance(sri_t.index, pd.DatetimeIndex):
            sri_t.index = pd.to_datetime(sri_t.index)
        sri_t.index = sri_t.index.to_period("M")

        pos_by_month = build_monthly_positions_from_sri(
            sri_t,
            leverage_risk_on=leverage_risk_on,
            leverage_risk_off=leverage_risk_off,
            risk_off_when_sri_is_one=risk_off_when_sri_is_one,
            default_leverage=default_leverage,
        )
        monthly_positions[tgt] = pos_by_month

        daily_pos = expand_monthly_positions_to_daily(r.index, pos_by_month, default_leverage=default_leverage)
        if fill_missing_with_zero:
            r = r.fillna(0.0)
        strat = daily_pos * r
        daily_leverage[tgt] = daily_pos
        strat_daily[tgt] = strat
        equity[tgt] = (1.0 + strat).cumprod()
        bh_equity[tgt] = (1.0 + r).cumprod()

    monthly_positions_df = pd.DataFrame(monthly_positions)
    if isinstance(monthly_positions_df.index, pd.PeriodIndex):
        monthly_positions_df.index = monthly_positions_df.index.to_timestamp(how="end")

    return {
        "daily_leverage": daily_leverage,
        "strategy_daily_returns": strat_daily,
        "equity_curves": equity,
        "buyhold_equity_curves": bh_equity,
        "monthly_positions": monthly_positions_df,
    }
