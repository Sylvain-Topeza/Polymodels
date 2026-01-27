from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestParams:
    """
    Monthly rebalancing backtest parameters.

    Convention (book-style):
    - Signal observed at month t (month-end) determines the leverage applied during month t+1.
    - We apply leverage on DAILY raw returns while holding it constant during the month.
    """
    leverage_risk_on: float = 2.0
    leverage_risk_off: float = 0.5

    # Interpret the SRI signal:
    # If True: SRI=1 means risk is high => use risk_off leverage
    # If False: SRI=1 means risk is low  => use risk_on leverage
    risk_off_when_sri_is_one: bool = True

    # How to handle missing positions / returns
    default_leverage: float = 1.0
    fill_missing_daily_returns_with_zero: bool = True


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def _extract_sri_wide(sri_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
    - DataFrame indexed by MultiIndex (date, market) with column ['sri']
    - Or a wide DataFrame indexed by date with market columns
    Returns wide df: index=date, columns=markets, values=sri in {0,1}.
    """
    if isinstance(sri_df.index, pd.MultiIndex):
        if sri_df.index.names != ["date", "market"]:
            sri_df = sri_df.copy()
            sri_df.index.set_names(["date", "market"], inplace=True)

        if "sri" not in sri_df.columns and sri_df.shape[1] == 1:
            sri_df = sri_df.rename(columns={sri_df.columns[0]: "sri"})

        wide = sri_df.reset_index().pivot(index="date", columns="market", values="sri")
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        return wide

    # Already wide
    wide = sri_df.copy()
    if "sri" in wide.columns and wide.shape[1] == 1:
        raise ValueError("sri_df looks like a single-column wide df; expected markets as columns.")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def _map_sri_to_leverage(
    sri_monthly: pd.Series,
    leverage_risk_on: float,
    leverage_risk_off: float,
    risk_off_when_sri_is_one: bool,
) -> pd.Series:
    """
    Map SRI in {0,1} to leverage.
    """
    s = sri_monthly.astype(float)
    if risk_off_when_sri_is_one:
        # SRI=1 => risk off
        return s.map(lambda v: leverage_risk_off if v >= 0.5 else leverage_risk_on)
    # SRI=1 => risk on
    return s.map(lambda v: leverage_risk_on if v >= 0.5 else leverage_risk_off)


def build_monthly_positions_from_sri(
    sri_monthly: pd.Series,
    params: BacktestParams,
) -> pd.Series:
    """
    Build monthly positions indexed by Period('M'), applying the 1-month lag:
    leverage for month p is determined by SRI observed at month p-1.

    Returns:
      positions_by_month: PeriodIndex('M') -> leverage
    """
    # Convert to month PeriodIndex
    if not isinstance(sri_monthly.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("sri_monthly index must be DatetimeIndex or PeriodIndex.")

    if isinstance(sri_monthly.index, pd.DatetimeIndex):
        sri_by_month = sri_monthly.copy()
        sri_by_month.index = sri_by_month.index.to_period("M")
    else:
        sri_by_month = sri_monthly.copy()

    lev_at_signal_month = _map_sri_to_leverage(
        sri_by_month,
        leverage_risk_on=params.leverage_risk_on,
        leverage_risk_off=params.leverage_risk_off,
        risk_off_when_sri_is_one=params.risk_off_when_sri_is_one,
    )

    # Apply lag: month p uses signal from p-1
    positions_by_month = lev_at_signal_month.shift(1)

    # Fill first month position (no prior signal)
    positions_by_month = positions_by_month.fillna(params.default_leverage)
    positions_by_month.name = "leverage"
    return positions_by_month


def expand_monthly_positions_to_daily(
    daily_index: pd.DatetimeIndex,
    positions_by_month: pd.Series,
    default_leverage: float = 1.0,
) -> pd.Series:
    """
    Expand monthly positions (PeriodIndex 'M') to a daily index.
    Each day gets the leverage of its month period.

    positions_by_month index must be PeriodIndex('M').
    """
    if not isinstance(daily_index, pd.DatetimeIndex):
        raise TypeError("daily_index must be a DatetimeIndex.")
    if not isinstance(positions_by_month.index, pd.PeriodIndex):
        raise TypeError("positions_by_month must have a PeriodIndex.")

    day_month = daily_index.to_period("M")
    # Reindex positions to include all months in daily data
    all_months = pd.PeriodIndex(day_month.unique(), freq="M").sort_values()
    pos_full = positions_by_month.reindex(all_months).fillna(default_leverage)

    daily_pos = pd.Series(pos_full.loc[day_month].to_numpy(), index=daily_index, name="leverage")
    return daily_pos


def run_monthly_backtest_single_market(
    daily_returns: pd.Series,
    sri_monthly: pd.Series,
    params: BacktestParams,
) -> Dict[str, pd.Series]:
    """
    Backtest for a single market:
      - monthly signal -> monthly leverage (lagged by 1 month)
      - leverage applied daily within each month

    Returns dict with:
      - daily_leverage
      - strategy_daily_returns
      - equity_curve
      - buyhold_equity_curve
    """
    r = daily_returns.astype(float).copy()
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()

    if params.fill_missing_daily_returns_with_zero:
        r = r.fillna(0.0)

    pos_by_month = build_monthly_positions_from_sri(sri_monthly, params=params)
    daily_lev = expand_monthly_positions_to_daily(r.index, pos_by_month, default_leverage=params.default_leverage)

    strat_ret = daily_lev * r
    strat_ret.name = "strategy_return"

    equity = (1.0 + strat_ret).cumprod()
    equity.name = "equity_curve"

    bh_equity = (1.0 + r).cumprod()
    bh_equity.name = "buyhold_equity_curve"

    return {
        "daily_leverage": daily_lev,
        "strategy_daily_returns": strat_ret,
        "equity_curve": equity,
        "buyhold_equity_curve": bh_equity,
    }


def run_monthly_backtest_multi_market(
    daily_returns_df: pd.DataFrame,
    sri_df: pd.DataFrame,
    markets: Sequence[str],
    params: BacktestParams,
) -> Dict[str, pd.DataFrame]:
    """
    Multi-market backtest.

    Inputs
    ------
    daily_returns_df:
      daily raw returns for ALL tickers; we subset to markets
    sri_df:
      output from Step 10 (MultiIndex (date, market), column 'sri') OR a wide monthly df

    Returns
    -------
    dict of DataFrames:
      - daily_leverage (daily index, columns markets)
      - strategy_daily_returns (daily index, columns markets)
      - equity_curves (daily index, columns markets)
      - buyhold_equity_curves (daily index, columns markets)
      - monthly_positions (PeriodIndex 'M', columns markets)
    """
    rets = _ensure_datetime_index(daily_returns_df)
    missing = [m for m in markets if m not in rets.columns]
    if missing:
        raise KeyError(f"Missing market columns in daily returns: {missing}")

    sri_wide = _extract_sri_wide(sri_df)
    # Ensure we have the requested markets in the SRI output
    missing_sri = [m for m in markets if m not in sri_wide.columns]
    if missing_sri:
        raise KeyError(f"Missing market columns in SRI: {missing_sri}")

    daily_leverage = pd.DataFrame(index=rets.index, columns=markets, dtype=float)
    strat_daily = pd.DataFrame(index=rets.index, columns=markets, dtype=float)
    equity = pd.DataFrame(index=rets.index, columns=markets, dtype=float)
    bh_equity = pd.DataFrame(index=rets.index, columns=markets, dtype=float)
    monthly_positions = {}

    for mkt in markets:
        r = rets[mkt]
        sri_m = sri_wide[mkt].dropna()

        # Build positions by month (PeriodIndex)
        if not isinstance(sri_m.index, pd.DatetimeIndex):
            sri_m.index = pd.to_datetime(sri_m.index)
        sri_m.index = sri_m.index.to_period("M")

        pos_by_month = build_monthly_positions_from_sri(sri_m, params=params)
        monthly_positions[mkt] = pos_by_month

        daily_pos = expand_monthly_positions_to_daily(r.index, pos_by_month, default_leverage=params.default_leverage)
        if params.fill_missing_daily_returns_with_zero:
            r = r.fillna(0.0)

        strat = daily_pos * r

        daily_leverage[mkt] = daily_pos
        strat_daily[mkt] = strat
        equity[mkt] = (1.0 + strat).cumprod()
        bh_equity[mkt] = (1.0 + r).cumprod()

    monthly_positions_df = pd.DataFrame(monthly_positions)
    return {
        "daily_leverage": daily_leverage,
        "strategy_daily_returns": strat_daily,
        "equity_curves": equity,
        "buyhold_equity_curves": bh_equity,
        "monthly_positions": monthly_positions_df,
    }