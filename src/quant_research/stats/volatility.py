"""Volatility estimators from close prices and from daily OHLC bars.

All estimators return a per-period *variance* series over a rolling window of ``window``
days; ``to_annual_vol`` converts to annualised volatility. Range-based estimators use the
information in the high and low, so they are more efficient (less noisy) than close-to-close
for the same window, under their assumptions.

| Estimator | Uses | Assumes |
|---|---|---|
| close-to-close | C | - |
| EWMA (RiskMetrics) | C | lambda = 0.94 daily |
| Parkinson (1980) | H, L | zero drift, no opening gap, continuous trading |
| Garman-Klass (1980) | O, H, L, C | zero drift, no opening gap |
| Rogers-Satchell (1991) | O, H, L, C | allows drift, no opening gap |
| Yang-Zhang (2000) | O, H, L, C, previous C | allows drift and opening gaps |
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant_research._typing import log, sqrt

LN2 = math.log(2.0)


def to_annual_vol[F: (pd.Series, pd.DataFrame)](variance: F, periods_per_year: int = 252) -> F:
    return sqrt(variance * periods_per_year)


def close_to_close(close: pd.Series, window: int = 21) -> pd.Series:
    r = log(close).diff()
    return r.rolling(window).var(ddof=1)


def ewma_variance(returns: pd.Series, lam: float = 0.94, init_window: int = 21) -> pd.Series:
    """RiskMetrics EWMA: s2_t = lam * s2_{t-1} + (1 - lam) * r_{t-1}^2.

    The value at date t uses returns up to t-1 only (a forecast for day t), so it can be used
    without look-ahead. Initialised with the sample variance of the first ``init_window``
    returns.
    """
    r = returns.dropna().to_numpy()
    out = np.full(r.size, np.nan)
    if r.size <= init_window:
        return pd.Series(out, index=returns.dropna().index)
    s2 = np.var(r[:init_window], ddof=1)
    for t in range(init_window, r.size):
        out[t] = s2  # forecast for t made with information up to t-1
        s2 = lam * s2 + (1 - lam) * r[t] ** 2
    return pd.Series(out, index=returns.dropna().index)


def parkinson(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    hl = log(high / low) ** 2
    return hl.rolling(window).mean() / (4.0 * LN2)


def garman_klass(
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,  # noqa: E741
    c: pd.Series,
    window: int = 21,
) -> pd.Series:
    term = 0.5 * log(h / l) ** 2 - (2.0 * LN2 - 1.0) * log(c / o) ** 2
    return term.rolling(window).mean()


def rogers_satchell(
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,  # noqa: E741
    c: pd.Series,
    window: int = 21,
) -> pd.Series:
    term = log(h / c) * log(h / o) + log(l / c) * log(l / o)
    return term.rolling(window).mean()


def yang_zhang(
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,  # noqa: E741
    c: pd.Series,
    window: int = 21,
) -> pd.Series:
    """sigma^2 = var(overnight) + k * var(open-to-close) + (1 - k) * RS,
    k = 0.34 / (1.34 + (n + 1) / (n - 1))."""
    overnight = log(o / c.shift(1))
    open_close = log(c / o)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    rs = rogers_satchell(o, h, l, c, window)
    return (
        overnight.rolling(window).var(ddof=1)
        + k * open_close.rolling(window).var(ddof=1)
        + (1 - k) * rs
    )


def all_estimators(bars: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Every estimator as annualised volatility, aligned on the bar dates."""
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]  # noqa: E741
    out = pd.DataFrame(
        {
            "close_to_close": close_to_close(c, window),
            "ewma": ewma_variance(log(c).diff()).reindex(bars.index),
            "parkinson": parkinson(h, l, window),
            "garman_klass": garman_klass(o, h, l, c, window),
            "rogers_satchell": rogers_satchell(o, h, l, c, window),
            "yang_zhang": yang_zhang(o, h, l, c, window),
        }
    )
    return to_annual_vol(out)
