from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal, Sequence, Optional, Dict, Any
import json
import hashlib


DataConstruction = Literal["prod"]  # per author/book: 21d compounding
TickerType = Literal["price", "rate"]


@dataclass(frozen=True)
class Config:
    """
    Central configuration for the SRI pipeline (Book ch.1-4).
    This config is designed to be notebook-friendly and fully serializable.
    """
    # --- Universe / targets ---
    markets: Sequence[str]  # list of market tickers to run SRI on (multi-market)
    factors: Optional[Sequence[str]] = None  # if None, use all non-market tickers as factors

    # --- Paths ---
    dictionary_path: str = "data/input/dictionary.xlsx"
    returns_path: str = "data/input/returns.csv"  # daily returns input for now

    artifacts_root: str = "data/artifacts"
    models_root: str = "models"
    results_root: str = "results"

    # --- Data construction ---
    data_construction: DataConstruction = "prod"
    rolling_window_days: int = 21  # "monthly returns available at daily frequency"

    # --- Estimation frequency ---
    estimation_frequency: Literal["monthly"] = "monthly"

    # --- LNLM estimation windows (monthly after sampling month-ends) ---
    window_months: int = 60  # 5 years
    min_window_months: int = 30  # allow warm-up if desired (otherwise set equal to window_months)
    lag_months: int = 1  # X_{t-1} -> Y_t

    # --- Pre-crisis memory ---
    forward_months: int = 3
    memory_halflife_years: float = 10.0

    # --- Clipping (optional stage) ---
    is_clipping: bool = False
    clip_assets: Optional[float] = None  # e.g. 0.20 for +/- 20%
    clip_rates: Optional[float] = None   # e.g. 0.01 (100 bps) in rate units
    clipping_mode: Literal["clip", "nan"] = "clip"

    # --- RMSE distribution ---
    kde_grid_size: int = 400
    kde_bandwidth: Optional[float] = None  # if None -> auto rule (later)

    # --- Backtest ---
    leverage_risk_on: float = 2.0
    leverage_risk_off: float = 0.5

    # --- Extra metadata ---
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def hash(self) -> str:
        """
        Stable hash of the configuration (used to version artifacts).
        """
        payload = self.to_json().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:10]
