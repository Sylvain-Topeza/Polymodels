"""
Single-factor LNLM fit, wrapping the external `lnlm.LinearNonlinearMixedRegressor`.

Reference: Barrau & Douady (2022) ch. 3.

The external library expects 1D numpy arrays. Cleanup of NaN and inf is the
caller's responsibility (handled in `compute_returns` and the panel
estimator).
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LnlmFitResult:
    """Serializable LNLM fit result for a single (target, feature, date) cell."""
    rmse: float
    r2: float
    mu: float
    y_mean: float
    linear_coef: float
    nonlinear_coef: Tuple[float, ...]
    n_obs: int
    degree_n: int
    n_folds: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["nonlinear_coef"] = list(self.nonlinear_coef)
        return d


# ---------------------------------------------------------------------------
# External library import
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    # src/models/lnlm_fit.py -> src/models -> src -> project_root
    return Path(__file__).resolve().parents[2]


def _import_external_lnlm():
    """Import external/lnlm.py without modifying it (it imports cross_validation by name)."""
    external_dir = _project_root() / "external"
    if not external_dir.exists():
        raise FileNotFoundError(f"External directory not found at: {external_dir}")
    ext_path = str(external_dir)
    if ext_path not in sys.path:
        sys.path.insert(0, ext_path)
    import lnlm  # type: ignore
    return lnlm


# ---------------------------------------------------------------------------
# Metrics (work on numpy arrays or pandas Series interchangeably)
# ---------------------------------------------------------------------------
def _rmse(y_true, y_pred) -> float:
    err = y_true - y_pred
    return float((err ** 2).mean() ** 0.5)


def _r2(y_true, y_pred) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    y_mean = y_true.mean() if hasattr(y_true, "mean") else float(np.mean(y_true))
    ss_tot = float(((y_true - y_mean) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_lnlm_single_factor(
    x: np.ndarray,
    y: np.ndarray,
    degree_n: int = 4,
    n_folds: int = 5,
    n_mu_points: int = 100,
    random_state: Optional[int] = 42,
) -> LnlmFitResult:
    """Fit one LNLM model on a single (feature, target) pair."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 10:
        raise ValueError("Not enough observations to fit LNLM.")

    lnlm = _import_external_lnlm()
    Model = lnlm.LinearNonlinearMixedRegressor  # type: ignore

    model = Model(
        mu="auto",
        n=degree_n,
        n_folds=n_folds,
        n_mu_points=n_mu_points,
        random_state=random_state,
    )

    X = x.reshape(-1, 1)
    model.fit(X, y)
    y_pred = model.predict(X)

    linear_coef = float(model.linear_model.coef_.reshape(-1)[0])  # type: ignore
    nonlinear_coef = tuple(float(v) for v in model.nonlinear_model.coef_.reshape(-1))  # type: ignore

    return LnlmFitResult(
        rmse=_rmse(y, y_pred),
        r2=_r2(y, y_pred),
        mu=float(model._mu_value),  # type: ignore
        y_mean=float(model.y_mean),  # type: ignore
        linear_coef=linear_coef,
        nonlinear_coef=nonlinear_coef,
        n_obs=int(len(x)),
        degree_n=int(degree_n),
        n_folds=int(n_folds),
    )
