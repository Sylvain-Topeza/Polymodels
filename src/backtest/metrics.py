"""
Performance metrics from a daily returns series.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    eq = equity.astype(float)
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    return float(dd.min())


def perf_metrics_from_daily_returns(
    daily_returns: pd.Series,
    equity_curve: Optional[pd.Series] = None,
    trading_days_per_year: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> Dict[str, float]:
    """Annualized return, vol, Sharpe, CAGR (corrected), max drawdown, Calmar."""
    r = daily_returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {}

    td = trading_days_per_year
    mean_d = float(r.mean())
    vol_d = float(r.std(ddof=1)) if r.size > 1 else 0.0
    ann_ret = (1.0 + mean_d) ** td - 1.0
    ann_vol = vol_d * np.sqrt(td)

    rf_d = (1.0 + risk_free_rate_annual) ** (1.0 / td) - 1.0
    excess_d = mean_d - rf_d
    sharpe = (excess_d / vol_d) * np.sqrt(td) if vol_d > 0 else float("nan")

    eq = equity_curve.astype(float) if equity_curve is not None else (1.0 + r).cumprod()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    # Corrected: use ratio of last to first equity (the previous version
    # assumed eq.iloc[0] == 1, which is only true when r[0] == 0).
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    mdd = _max_drawdown(eq)
    calmar = (-cagr / mdd) if mdd < 0 else float("nan")

    return {
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "max_drawdown": float(mdd),
        "calmar": float(calmar),
        "final_equity": float(eq.iloc[-1]),
    }


def perf_metrics_multi_target(
    strategy_daily_returns: pd.DataFrame,
    equity_curves: pd.DataFrame,
    trading_days_per_year: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """Per-target metrics from wide return / equity DataFrames."""
    return {
        str(col): perf_metrics_from_daily_returns(
            strategy_daily_returns[col],
            equity_curve=equity_curves[col],
            trading_days_per_year=trading_days_per_year,
            risk_free_rate_annual=risk_free_rate_annual,
        )
        for col in strategy_daily_returns.columns
    }
