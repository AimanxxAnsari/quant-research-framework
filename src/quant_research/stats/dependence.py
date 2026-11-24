"""Correlation, beta and lead-lag structure, including non-synchronous data effects.

When two series are sampled at different times of day (e.g. an FX noon fixing vs a 4 pm equity
close), part of their co-movement shows up one day later. Contemporaneous daily correlation is
then biased towards zero, and the lag-1 cross-correlation becomes non-zero. Aggregating to
weekly returns recovers most of the true correlation (Burns, Engle & Mezrich 1998).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def beta(y: pd.Series, x: pd.Series) -> dict:
    """OLS slope of y on x with intercept, standard error and R^2."""
    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    xv = df["x"].to_numpy()
    yv = df["y"].to_numpy()
    xm, ym = xv - xv.mean(), yv - yv.mean()
    b = (xm @ ym) / (xm @ xm)
    resid = ym - b * xm
    n = len(df)
    se = np.sqrt((resid @ resid) / (n - 2) / (xm @ xm))
    r2 = 1 - (resid @ resid) / (ym @ ym)
    return {"beta": float(b), "se": float(se), "r2": float(r2), "n": n}


def rolling_beta(y: pd.Series, x: pd.Series, window: int = 63) -> pd.Series:
    cov = y.rolling(window).cov(x)
    return cov / x.rolling(window).var()


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    return a.rolling(window).corr(b)


def lead_lag_corr(x: pd.Series, y: pd.Series, lags: range = range(-3, 4)) -> pd.Series:
    """corr(x_t, y_{t+k}) for each lag k. A peak at k > 0 means y reacts after x."""
    out = {}
    for k in lags:
        pair = pd.concat([x, y.shift(-k)], axis=1).dropna()
        out[k] = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return pd.Series(out, name="corr").rename_axis("lag")


def sum_of_lags_corr(x: pd.Series, y: pd.Series, max_lag: int = 1) -> float:
    """Non-synchronous-trading correction in the spirit of Scholes-Williams (1977) and Dimson
    (1979): sum the cross-covariances cov(x_t, y_{t+k}) for k = -L..L and scale by the
    single-period standard deviations. Recovers the correlation that daily sampling splits
    across adjacent days."""
    pair = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    xd = pair["x"] - pair["x"].mean()
    yd = pair["y"] - pair["y"].mean()
    cov = sum((xd * yd.shift(-k)).mean() for k in range(-max_lag, max_lag + 1))
    return float(cov / (xd.std(ddof=0) * yd.std(ddof=0)))
