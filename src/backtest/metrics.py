from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricsParams:
    trading_days_per_year: int = 252
    risk_free_rate_annual: float = 0.0


def _max_drawdown(equity: pd.Series) -> float:
    eq = equity.astype(float)
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    return float(dd.min())


def perf_metrics_from_daily_returns(
    daily_returns: pd.Series,
    equity_curve: Optional[pd.Series] = None,
    params: MetricsParams = MetricsParams(),
) -> Dict[str, float]:
    """
    Compute common performance metrics from daily returns.
    """
    r = daily_returns.astype(float).copy()
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {}

    td = params.trading_days_per_year
    mean_daily = float(r.mean())
    vol_daily = float(r.std(ddof=1)) if r.size > 1 else 0.0

    ann_ret = (1.0 + mean_daily) ** td - 1.0
    ann_vol = vol_daily * np.sqrt(td)

    rf_daily = (1.0 + params.risk_free_rate_annual) ** (1.0 / td) - 1.0
    excess_daily = mean_daily - rf_daily
    sharpe = (excess_daily / vol_daily) * np.sqrt(td) if vol_daily > 0 else float("nan")

    # CAGR from equity if provided
    if equity_curve is None:
        eq = (1.0 + r).cumprod()
    else:
        eq = equity_curve.astype(float)

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")

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


def perf_metrics_multi_market(
    strategy_daily_returns: pd.DataFrame,
    equity_curves: pd.DataFrame,
    params: MetricsParams = MetricsParams(),
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each market column.
    """
    out: Dict[str, Dict[str, float]] = {}
    for col in strategy_daily_returns.columns:
        metrics = perf_metrics_from_daily_returns(
            strategy_daily_returns[col],
            equity_curve=equity_curves[col],
            params=params,
        )
        out[str(col)] = metrics
    return out