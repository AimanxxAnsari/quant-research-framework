"""Random-walk diagnostics: variance ratio and Hurst exponent.

Variance ratio (Lo & MacKinlay 1988)
------------------------------------
For log returns r_1..r_T, VR(q) = Var(q-period return) / (q * Var(1-period return)).
Under a random walk VR(q) = 1; VR > 1 suggests trending (positive autocorrelation), VR < 1
mean reversion. In population, VR(2) = 1 + rho_1. Uses overlapping q-period returns with the
paper's bias corrections, and reports both the homoskedastic z and the
heteroskedasticity-robust z* (robust to volatility clustering, which real returns have).

Hurst exponent
--------------
Classical rescaled range (R/S): E[R/S(n)] ~ c * n^H; H = 0.5 for a random walk, > 0.5
persistent, < 0.5 anti-persistent. The estimate is biased upward in small samples and is also
pushed up by *short*-range autocorrelation, so it is compared with a shuffled baseline.
Lo's (1991) modified R/S statistic V corrects for short-range dependence; under the null of no
long memory, V lies in [0.809, 1.862] with 95% probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from quant_research.errors import InvalidParameterError

LO_95_INTERVAL = (0.809, 1.862)


@dataclass(frozen=True)
class VarianceRatio:
    q: int
    vr: float
    z: float  # homoskedastic
    z_star: float  # heteroskedasticity-robust
    p_value: float  # two-sided, from z_star


def variance_ratio(returns: pd.Series | np.ndarray, q: int) -> VarianceRatio:
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    T = r.size
    if q < 2 or q >= T:
        raise InvalidParameterError("need 2 <= q < number of returns")
    mu = r.mean()
    d = r - mu
    var_a = (d @ d) / (T - 1)
    p = np.concatenate([[0.0], np.cumsum(r)])
    q_ret = p[q:] - p[:-q] - q * mu  # overlapping q-period returns, demeaned
    m = q * (T - q + 1) * (1 - q / T)
    var_c = (q_ret @ q_ret) / m
    vr = var_c / var_a

    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    d2 = d**2
    denom = (d2.sum()) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = float(T * (d2[j:] @ d2[:-j]) / denom)
        theta += (2 * (q - j) / q) ** 2 * delta
    z_star = (vr - 1) / np.sqrt(theta / T)  # theta is the asymptotic variance of sqrt(T)(VR-1)
    return VarianceRatio(
        q, float(vr), float(z), float(z_star), float(2 * stats.norm.sf(abs(z_star)))
    )


def _rs(x: np.ndarray) -> float:
    y = np.cumsum(x - x.mean())
    s = x.std(ddof=0)
    return (y.max() - y.min()) / s if s > 0 else np.nan


def hurst_rs(returns: pd.Series | np.ndarray, min_n: int = 16, n_sizes: int = 12) -> float:
    """Classical R/S Hurst exponent: slope of log mean R/S(n) on log n over non-overlapping
    blocks of log-spaced sizes n."""
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    max_n = x.size // 4
    sizes = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), n_sizes).astype(int))
    pts = []
    for n in sizes:
        k = x.size // n
        rs = [_rs(x[i * n : (i + 1) * n]) for i in range(k)]
        pts.append((np.log(n), np.log(np.nanmean(rs))))
    a = np.array(pts)
    return float(np.polyfit(a[:, 0], a[:, 1], 1)[0])


def hurst_shuffled_baseline(
    returns: pd.Series | np.ndarray, n_shuffles: int = 20, seed: int = 0
) -> tuple[float, float]:
    """Mean and std of the R/S Hurst estimate on shuffled data (no memory by construction).
    The real estimate is only interesting relative to this baseline."""
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    rng = np.random.default_rng(seed)
    hs = [hurst_rs(rng.permutation(x)) for _ in range(n_shuffles)]
    return float(np.mean(hs)), float(np.std(hs, ddof=1))


def lo_modified_rs(
    returns: pd.Series | np.ndarray, q: int | None = None
) -> dict[str, float | int | bool]:
    """Lo (1991) modified R/S statistic V with Andrews (1991) automatic bandwidth."""
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    T = x.size
    d = x - x.mean()
    if q is None:
        rho = np.corrcoef(d[1:], d[:-1])[0, 1]
        rho = np.clip(rho, -0.99, 0.99)
        q = (
            int(np.floor((1.5 * T) ** (1 / 3) * (2 * rho / (1 - rho**2)) ** (2 / 3)))
            if rho > 0
            else 0
        )
    gamma0 = (d @ d) / T
    s2 = gamma0
    for j in range(1, q + 1):
        s2 += 2 * (1 - j / (q + 1)) * (d[j:] @ d[:-j]) / T
    y = np.cumsum(d)
    v = (y.max() - y.min()) / np.sqrt(s2) / np.sqrt(T)
    lo, hi = LO_95_INTERVAL
    return {"V": float(v), "q": q, "reject_no_long_memory": not (lo <= v <= hi)}
