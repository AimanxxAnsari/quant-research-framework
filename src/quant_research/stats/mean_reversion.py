"""Mean-reversion estimators: AR(1) half-life and Engle-Granger cointegration.

Half-life: fit x_t = c + phi x_{t-1} + e_t by OLS; half-life = ln(1/2) / ln(phi) for 0 < phi < 1.
OLS phi is biased downwards in small samples (Kendall 1954: E[phi_hat] - phi ~ -(1 + 3 phi)/n),
so half-lives are underestimated: mean reversion looks faster than it is.

Engle-Granger (1987): regress y on x in levels, then test the residual for a unit root with an
ADF test using cointegration critical values (MacKinnon). Independent random walks regressed
on each other give spuriously high R^2 and t-statistics (Granger & Newbold 1974); only the
residual unit-root test is valid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.tsa.stattools import coint


def ar1_fit(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y, lag = x[1:], x[:-1]
    X = np.column_stack([np.ones_like(lag), lag])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    phi = float(coef[1])
    hl = float(np.log(0.5) / np.log(phi)) if 0 < phi < 1 else float("inf")
    return {"c": float(coef[0]), "phi": phi, "half_life": hl}


def ols_levels(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """Levels regression y = a + b x: slope, its naive t-statistic and R^2 (spurious if both
    series are independent random walks)."""
    X = np.column_stack([np.ones_like(x), x])
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n = len(y)
    s2 = resid @ resid / (n - 2)
    xm = x - x.mean()
    se_b = np.sqrt(s2 / (xm @ xm))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return {"beta": float(coef[1]), "t": float(coef[1] / se_b), "r2": float(r2)}


def engle_granger(y: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    """Engle-Granger test (statsmodels ``coint``, constant trend) plus the levels hedge ratio."""
    stat, p, _ = coint(y, x, trend="c", autolag=None, maxlag=1)
    return {"t_stat": float(stat), "p_value": float(p), "beta": ols_levels(y, x)["beta"]}
