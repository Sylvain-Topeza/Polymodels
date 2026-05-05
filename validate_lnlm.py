"""
Validation of the external LNLM library:
  1. Verify visually that LinearNonlinearMixedRegressor produces a sensible
     curve sitting between a pure linear fit and a pure nonlinear fit.
  2. Verify that fitting on ~1000 points takes seconds, not minutes.
  3. Optionally, line_profiler highlights the hottest lines for further work.

Outputs:
  - lnlm_check_100.png
  - lnlm_check_1000.png
  - timing summary in stdout
  - optional line_profiler stats if line_profiler is installed
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Make the external lnlm.py importable
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
EXTERNAL_DIR = HERE / "external"
if not EXTERNAL_DIR.exists():
    raise FileNotFoundError(
        f"Expected external/ directory at {EXTERNAL_DIR}. "
        "If your layout differs, edit HERE / EXTERNAL_DIR at the top of this file."
    )
sys.path.insert(0, str(EXTERNAL_DIR))

from lnlm import LinearNonlinearMixedRegressor  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data with a known nonlinear signal
# ---------------------------------------------------------------------------
def generate_synthetic(n: int, noise: float = 0.4, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """y = 0.4 x + 0.25 x^2 - 0.05 x^3 + Gaussian noise.

    A modest cubic-with-quadratic signal so neither pure linear nor pure
    polynomial wins by default. The 'auto' fit should land between them.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n)
    y = 0.4 * x + 0.25 * x**2 - 0.05 * x**3 + noise * rng.standard_normal(n)
    return x, y


# ---------------------------------------------------------------------------
# Visual sanity check
# ---------------------------------------------------------------------------
def fit_three_variants(x: np.ndarray, y: np.ndarray) -> Dict[str, LinearNonlinearMixedRegressor]:
    """Fit LNLM with mu='auto', then forced mu=0 (linear) and mu=1 (polynomial)."""
    X = x.reshape(-1, 1)
    fits: Dict[str, LinearNonlinearMixedRegressor] = {}
    for label, mu in [("LNLM (mu='auto')", "auto"), ("linear (mu=0)", 0.0), ("nonlinear (mu=1)", 1.0)]:
        m = LinearNonlinearMixedRegressor(mu=mu, n=4, n_folds=5, n_mu_points=100, random_state=42)
        m.fit(X, y)
        fits[label] = m
    return fits


def plot_curves(x: np.ndarray, y: np.ndarray, fits: Dict[str, LinearNonlinearMixedRegressor], save_path: str) -> None:
    grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 400)
    Xg = grid.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, y, s=14, alpha=0.45, color="#666", label=f"data ({len(x)} points)")
    for label, m in fits.items():
        pred = m.predict(Xg)
        suffix = f"  [estimated mu = {m._mu_value:.3f}]" if "auto" in label else ""
        ax.plot(grid, pred, lw=2, label=label + suffix)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"LNLM sanity check on {len(x)} synthetic points")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  saved figure: {save_path}")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_fit(n: int, repeats: int = 3) -> float:
    """Best-of-N wall time for a full fit on n points."""
    x, y = generate_synthetic(n)
    X = x.reshape(-1, 1)
    durations = []
    for _ in range(repeats):
        m = LinearNonlinearMixedRegressor(mu="auto", n=4, n_folds=5, n_mu_points=100, random_state=42)
        t0 = time.perf_counter()
        m.fit(X, y)
        durations.append(time.perf_counter() - t0)
    return min(durations)


# ---------------------------------------------------------------------------
# Optional line profiling
# ---------------------------------------------------------------------------
def profile_with_line_profiler() -> None:
    """Identify hot lines in the fit pipeline. Requires `pip install line_profiler`."""
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print("  line_profiler not installed, skipping. Run: pip install line_profiler")
        return

    x, y = generate_synthetic(1000)
    X = x.reshape(-1, 1)
    m = LinearNonlinearMixedRegressor(mu="auto", n=4, n_folds=5, n_mu_points=100, random_state=42)

    lp = LineProfiler()
    lp.add_function(m._estimate_mu)
    lp.add_function(m._compute_hermite_features)
    lp.add_function(m._compute_rmse)
    lp.add_function(m._optimize_fold)

    wrapped = lp(m.fit)
    wrapped(X, y)
    lp.print_stats()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("PHASE 0 - LNLM VALIDATION")
    print("=" * 70)

    print("\n1. Visual sanity check on 100 points")
    x, y = generate_synthetic(100)
    fits = fit_three_variants(x, y)
    plot_curves(x, y, fits, save_path="lnlm_check_100.png")

    print("\n2. Visual sanity check on 1000 points")
    x, y = generate_synthetic(1000)
    fits = fit_three_variants(x, y)
    plot_curves(x, y, fits, save_path="lnlm_check_1000.png")

    print("\n3. Timing benchmark")
    targets = {100: 1.0, 500: 5.0, 1000: 15.0}  # acceptable seconds per fit
    for n, budget in targets.items():
        t = time_fit(n, repeats=3)
        verdict = "OK" if t <= budget else "SLOW"
        print(f"  n={n:>4}  best of 3 = {t:7.3f}s   budget {budget}s   [{verdict}]")

    print("\n4. Optional line profiling on 1000 points")
    profile_with_line_profiler()

    print("\n" + "=" * 70)
    print("DONE - inspect lnlm_check_100.png and lnlm_check_1000.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
