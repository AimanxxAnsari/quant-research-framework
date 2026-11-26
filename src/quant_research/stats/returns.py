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

from typing import Any, cast, overload

import numpy as np
import pandas as pd

from quant_research._typing import log
from quant_research.errors import EmptySeriesError

PERIODS = {"daily": 252, "weekly": 52, "monthly": 12}


@overload
def simple_returns(prices: pd.Series) -> pd.Series: ...
@overload
def simple_returns(prices: pd.DataFrame) -> pd.DataFrame: ...
def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change(fill_method=None).iloc[1:]


@overload
def log_returns(prices: pd.Series) -> pd.Series: ...
@overload
def log_returns(prices: pd.DataFrame) -> pd.DataFrame: ...
def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return log(prices).diff().iloc[1:]


def cumulative_return(simple: pd.Series) -> float:
    """Total compounded return from a series of simple returns."""
    r = simple.dropna().to_numpy(dtype=float)
    return float(np.prod(1.0 + r)) - 1.0


def cagr(simple: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from simple returns."""
    r = simple.dropna().to_numpy(dtype=float)
    if r.size == 0:
        raise EmptySeriesError("cagr needs at least one return")
    growth = float(np.prod(1.0 + r))
    return float(growth ** (periods_per_year / r.size)) - 1.0


def annualised_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


@overload
def resample_prices(prices: pd.Series, rule: str = ...) -> pd.Series: ...
@overload
def resample_prices(prices: pd.DataFrame, rule: str = ...) -> pd.DataFrame: ...
def resample_prices(
    prices: pd.DataFrame | pd.Series, rule: str = "W-WED"
) -> pd.DataFrame | pd.Series:
    """Last price in each period. Weekly default ends on Wednesday, which avoids most
    holidays (many fall on Mondays and Fridays)."""
    return prices.resample(rule).last()


def drawdown(prices: pd.Series) -> pd.Series:
    """Drawdown from the running peak, as a non-positive fraction."""
    p = prices.dropna()
    return p / p.cummax() - 1.0


def max_drawdown(prices: pd.Series) -> dict[str, Any]:
    """Depth, peak date, trough date and recovery date (None if not yet recovered)."""
    dd = drawdown(prices)
    trough = cast(pd.Timestamp, dd.idxmin())
    peak = cast(pd.Timestamp, prices.loc[:trough].idxmax())
    after = prices.loc[trough:]
    recovered = after[after >= prices.loc[peak]]
    return {
        "max_drawdown": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "recovery": recovered.index[0] if len(recovered) else None,
    }
