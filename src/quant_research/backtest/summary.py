"""Headline numbers for a backtest, and a side-by-side table across runs.

A first set for comparing engines and fill rules. Pillar 4 step 3 adds the full metric set
(Sortino, Calmar, hit rate, exposure) and the statistics that correct for multiple testing.

Conventions: simple daily returns; annualised with 252 trading days for volatility and Sharpe,
and with calendar time (365.25 days a year) for CAGR, cost drag and turnover. Sharpe uses
returns in excess of USD cash, since an all-cash portfolio earns the cash rate.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from quant_research.backtest.market import Market
from quant_research.backtest.result import BacktestResult

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25


def summary(result: BacktestResult, market: Market) -> dict[str, float]:
    nav = result.nav
    r = result.returns
    cash_growth, _ = market.accrual_factors()
    excess = r.to_numpy() - (cash_growth[1:] - 1.0)
    years = (nav.index[-1] - nav.index[0]).days / DAYS_PER_YEAR
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    sd = float(np.std(excess, ddof=1))
    drawdown = nav / nav.cummax() - 1.0
    cost_frac = result.costs.to_numpy()[1:] / nav.to_numpy()[:-1]
    return {
        "total return": total,
        "CAGR": float((1.0 + total) ** (1.0 / years) - 1.0),
        "volatility": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "Sharpe (excess of cash)": float(excess.mean() / sd * np.sqrt(TRADING_DAYS))
        if sd > 0
        else 0.0,
        "max drawdown": float(drawdown.min()),
        "cost drag per year": float(cost_frac.sum() / years),
        "turnover per year": float(result.turnover.sum() / years),
    }


def side_by_side(results: Mapping[str, BacktestResult], market: Market) -> pd.DataFrame:
    """One column per run, one row per headline number."""
    return pd.DataFrame({name: summary(res, market) for name, res in results.items()})
