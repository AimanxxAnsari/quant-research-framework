"""Value at Risk forecasts and the Kupiec proportion-of-failures backtest.

VaR is reported as a positive loss threshold: a breach occurs when the return is below -VaR.
All forecasts for day t use information up to t-1 only.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def historical_var(returns: np.ndarray, level: float = 0.99, window: int = 250) -> np.ndarray:
    """Rolling empirical quantile of the previous ``window`` returns."""
    r = np.asarray(returns, dtype=float)
    out = np.full(r.size, np.nan)
    for t in range(window, r.size):
        out[t] = -np.quantile(r[t - window : t], 1 - level)
    return out


def ewma_normal_var(
    returns: np.ndarray, level: float = 0.99, lam: float = 0.94, init: int = 250
) -> np.ndarray:
    """RiskMetrics: EWMA volatility forecast times the normal quantile."""
    r = np.asarray(returns, dtype=float)
    out = np.full(r.size, np.nan)
    s2 = np.var(r[:init], ddof=1)
    z = stats.norm.ppf(level)
    for t in range(init, r.size):
        out[t] = z * np.sqrt(s2)
        s2 = lam * s2 + (1 - lam) * r[t] ** 2
    return out


def breaches(returns: np.ndarray, var: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(var)
    return (np.asarray(returns)[mask] < -var[mask]).astype(int)


def kupiec_pof(n_breaches: int, n_obs: int, level: float = 0.99) -> dict[str, float]:
    """Kupiec (1995) LR test that the breach rate equals 1 - level (chi-square, 1 df)."""
    p = 1 - level
    x, T = n_breaches, n_obs
    phat = x / T

    def ll(q: float) -> float:
        out = 0.0
        if T - x > 0:
            out += (T - x) * np.log(1 - q)
        if x > 0:
            out += x * np.log(q)
        return out

    lr = -2 * (ll(p) - (ll(phat) if 0 < phat < 1 else 0.0))
    return {"breach_rate": phat, "lr": float(lr), "p_value": float(stats.chi2.sf(lr, 1))}
