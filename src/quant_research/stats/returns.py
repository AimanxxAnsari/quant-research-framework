"""Returns, compounding, annualisation and drawdowns.

Conventions
-----------
* simple return  R_t = P_t / P_{t-1} - 1   (aggregates across assets: portfolio = sum w_i R_i)
* log return     r_t = ln(P_t / P_{t-1})   (aggregates across time: r_{0,T} = sum r_t)
* r = ln(1 + R); for small moves r ~ R, but they diverge for large moves.
* Annualising a mean of log returns multiplies by the periods per year; annualising a standard
  deviation multiplies by sqrt(periods per year), which assumes independent returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = {"daily": 252, "weekly": 52, "monthly": 12}


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.log(prices).diff().iloc[1:]


def cumulative_return(simple: pd.Series) -> float:
    """Total compounded return from a series of simple returns."""
    return float((1.0 + simple.dropna()).prod() - 1.0)


def cagr(simple: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from simple returns."""
    r = simple.dropna()
    return float((1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0)


def annualised_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


def resample_prices(prices: pd.DataFrame | pd.Series, rule: str = "W-WED"):
    """Last price in each period. Weekly default ends on Wednesday, which avoids most
    holidays (many fall on Mondays and Fridays)."""
    return prices.resample(rule).last()


def drawdown(prices: pd.Series) -> pd.Series:
    """Drawdown from the running peak, as a non-positive fraction."""
    p = prices.dropna()
    return p / p.cummax() - 1.0


def max_drawdown(prices: pd.Series) -> dict:
    """Depth, peak date, trough date and recovery date (None if not yet recovered)."""
    dd = drawdown(prices)
    trough = dd.idxmin()
    peak = prices.loc[:trough].idxmax()
    after = prices.loc[trough:]
    recovered = after[after >= prices.loc[peak]]
    return {
        "max_drawdown": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "recovery": recovered.index[0] if len(recovered) else None,
    }
