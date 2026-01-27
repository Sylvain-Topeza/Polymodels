from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LnlmFitResult:
    """
    Serializable LNLM fit result for a single (market, factor, date) estimation.
    """
    rmse: float
    r2: float
    mu: float
    y_mean: float

    # coefficients
    linear_coef: float  # single factor => scalar
    nonlinear_coef: Tuple[float, ...]  # length = n (Hermite degrees)

    # diagnostics / metadata
    n_obs: int
    degree_n: int
    n_folds: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["nonlinear_coef"] = list(self.nonlinear_coef)
        return d


def _get_project_root() -> Path:
    """
    src/models/lnlm_fit.py -> src/models -> src -> project_root
    """
    return Path(__file__).resolve().parents[2]


def _import_external_lnlm():
    """
    Import external lnlm without modifying external code.

    external/lnlm.py imports 'cross_validation' without package prefix, so
    the external directory must be on sys.path.
    """
    import sys
    root = _get_project_root()
    external_dir = root / "external"
    if not external_dir.exists():
        raise FileNotFoundError(f"External directory not found at: {external_dir}")

    ext_path = str(external_dir)
    if ext_path not in sys.path:
        sys.path.insert(0, ext_path)

    # Now these imports resolve:
    import lnlm  # type: ignore
    return lnlm


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_lnlm_single_factor(
    x: np.ndarray,
    y: np.ndarray,
    degree_n: int = 4,
    n_folds: int = 5,
    n_mu_points: int = 100,
    random_state: Optional[int] = None,
) -> LnlmFitResult:
    """
    Fit a LinearNonlinearMixedRegressor on one factor and one market series.

    Notes:
    - Uses external implementation (do not edit external/lnlm.py).
    - RMSE and R2 computed on training sample (in-sample), consistent with current codebase.
    - Stores linear and nonlinear coefficients for refit / audit.

    Parameters
    ----------
    x : array shape (n_obs,)
    y : array shape (n_obs,)
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 10:
        raise ValueError("Not enough observations to fit LNLM.")

    # Drop non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 10:
        raise ValueError("Not enough finite observations after filtering.")

    lnlm = _import_external_lnlm()
    Model = lnlm.LinearNonlinearMixedRegressor  # type: ignore

    model = Model(
        mu=mu,
        n=degree_n,
        n_folds=n_folds,
        n_mu_points=n_mu_points,
        random_state=random_state,
    )

    X = x.reshape(-1, 1)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Extract coefficients
    # For 1 factor:
    linear_coef = float(model.linear_model.coef_.reshape(-1)[0])  # type: ignore
    nonlinear_coef = tuple(float(v) for v in model.nonlinear_model.coef_.reshape(-1))  # type: ignore

    return LnlmFitResult(
        rmse=_rmse(y, y_pred),
        r2=_r2(y, y_pred),
        mu=float(model._mu_value),   # type: ignore
        y_mean=float(model.y_mean),  # type: ignore
        linear_coef=linear_coef,
        nonlinear_coef=nonlinear_coef,
        n_obs=int(len(x)),
        degree_n=int(degree_n),
        n_folds=int(n_folds),
    )