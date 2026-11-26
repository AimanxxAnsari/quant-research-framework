"""Performance metrics for a backtest, a side-by-side table, and the cost sweep.

Conventions (ADR-0009 and ADR-0011):

* simple daily returns of NAV; "excess" means minus the USD cash return of the same day;
* volatility, Sharpe and Sortino annualised with 252 trading days and computed on excess
  returns, since an all-cash portfolio earns the cash rate and should score zero;
* Sortino uses the downside deviation sqrt(mean(min(excess, 0)^2)) over all days;
* CAGR, cost drag and turnover per calendar year (365.25 days);
* Calmar = CAGR / |max drawdown|;
* exposure = share of days with any position; hit rate = share of days with a position on
  which the excess return was positive; average gross leverage = mean of sum |weights|.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quant_research.backtest.config import BacktestConfig
from quant_research.backtest.market import Market
from quant_research.backtest.result import BacktestResult
from quant_research.backtest.runner import Engine, run
from quant_research.backtest.strategy import Strategy

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25


def excess_returns(result: BacktestResult, market: Market) -> np.ndarray:
    """Daily NAV returns minus the USD cash return over the same day."""
    cash_growth, _ = market.accrual_factors()
    return result.returns.to_numpy() - (cash_growth[1:] - 1.0)


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def summary(result: BacktestResult, market: Market) -> dict[str, float]:
    nav = result.nav
    r = result.returns.to_numpy()
    ex = excess_returns(result, market)
    years = (nav.index[-1] - nav.index[0]).days / DAYS_PER_YEAR
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
    max_dd = float((nav / nav.cummax() - 1.0).min())
    gross = result.weights.abs().sum(axis=1).to_numpy()
    invested = gross[:-1] > 0  # position held into the next day
    downside = float(np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2)))
    ann = np.sqrt(TRADING_DAYS)
    return {
        "total return": total,
        "CAGR": cagr,
        "volatility": float(r.std(ddof=1) * ann),
        "Sharpe (excess of cash)": _ratio(float(ex.mean()), float(ex.std(ddof=1))) * ann,
        "Sortino": _ratio(float(ex.mean()), downside) * ann,
        "max drawdown": max_dd,
        "Calmar": _ratio(cagr, abs(max_dd)),
        "hit rate": float(np.mean(ex[invested] > 0)) if invested.any() else 0.0,
        "exposure": float(np.mean(invested)),
        "average gross leverage": float(gross.mean()),
        "turnover per year": float(result.turnover.sum() / years),
        "cost drag per year": float(
            (result.costs.to_numpy()[1:] / nav.to_numpy()[:-1]).sum() / years
        ),
    }


def side_by_side(results: Mapping[str, BacktestResult], market: Market) -> pd.DataFrame:
    """One column per run, one row per metric."""
    return pd.DataFrame({name: summary(res, market) for name, res in results.items()})


def cost_sweep(
    strategy: Strategy,
    market: Market,
    config: BacktestConfig,
    multipliers: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    engine: Engine = "weights",
) -> pd.DataFrame:
    """Metrics with every cost scaled by each multiplier (0x shows the cost-free result)."""
    runs = {}
    for m in multipliers:
        scaled = BacktestConfig(
            fill=config.fill,
            initial_nav=config.initial_nav,
            costs=config.costs.scaled(m),
            max_gross_leverage=config.max_gross_leverage,
            lookahead_checks=config.lookahead_checks,
        )
        runs[f"costs x{m:g}"] = run(strategy, market, scaled, engine)
    return side_by_side(runs, market)
