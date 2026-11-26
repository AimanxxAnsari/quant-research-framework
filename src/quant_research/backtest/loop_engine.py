"""The bar-by-bar engine: slow, explicit, and unable to look ahead (ADR-0010).

It keeps the portfolio as a USD cash balance and a number of units of each asset, and walks
through the dates one at a time. On each date t:

1. interest: cash and FX balances grow by one period of their short rate;
2. next-open fills: yesterday's decision is traded at today's open;
3. next-close fills: yesterday's decision is traded at today's close;
4. NAV is marked at today's close;
5. the strategy is shown the market *only up to today's close* and returns today's decision.

Because the strategy never receives future rows, a decision cannot depend on them. It uses
the same conventions as the weights engine but an independent implementation (units and cash
rather than weights and growth factors), so the two cross-check each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.backtest.config import BacktestConfig
from quant_research.backtest.market import Market
from quant_research.backtest.result import BacktestResult
from quant_research.backtest.strategy import Strategy
from quant_research.backtest.targets import align_targets, check_targets
from quant_research.errors import AccountWipedOutError, MissingValuesError

ENGINE = "loop"


class _Book:
    """Cash in USD and units held of each asset."""

    def __init__(self, cash: float, n_assets: int, cost_fractions: np.ndarray) -> None:
        self.cash = cash
        self.units = np.zeros(n_assets)
        self.k = cost_fractions

    def value(self, prices: np.ndarray) -> np.ndarray:
        held = self.units != 0
        if np.isnan(prices[held]).any():
            raise MissingValuesError("a held asset has no price")
        return np.where(held, self.units * np.nan_to_num(prices), 0.0)

    def nav(self, prices: np.ndarray) -> float:
        return float(self.cash + self.value(prices).sum())

    def accrue(self, cash_growth: float, asset_growth: np.ndarray) -> None:
        self.cash *= cash_growth
        self.units = self.units * asset_growth

    def trade_to(
        self, weights: np.ndarray, prices: np.ndarray, date: pd.Timestamp
    ) -> tuple[float, float]:
        """Trade to target weights of the NAV at these prices. Returns (cost, turnover)."""
        nav = self.nav(prices)
        if nav <= 0:
            raise AccountWipedOutError(f"net asset value fell to zero or below on {date:%Y-%m-%d}")
        px = np.nan_to_num(prices, nan=1.0)  # only where both target and holding are zero
        target = np.where(weights != 0, weights * nav / px, 0.0)
        delta = target - self.units
        notional = np.abs(delta) * px
        cost = float((notional * self.k).sum())
        self.cash -= float((delta * px).sum()) + cost
        self.units = target
        return cost, float(notional.sum() / nav)


def run(strategy: Strategy, market: Market, config: BacktestConfig) -> BacktestResult:
    dates = market.dates
    close = market.close.to_numpy(dtype=float)
    opens = market.open.to_numpy(dtype=float)
    cash_growth, asset_growth = market.accrual_factors()
    book = _Book(
        config.initial_nav,
        len(market.assets),
        config.costs.fractions([market.asset_class[a] for a in market.assets]),
    )
    n = len(dates)
    nav, costs, turnover = np.zeros(n), np.zeros(n), np.zeros(n)
    values = np.zeros((n, len(market.assets)))
    pending: np.ndarray | None = None

    for t in range(n):
        if t > 0:
            book.accrue(float(cash_growth[t]), asset_growth[t])
        if pending is not None:
            px = opens[t] if config.fill == "next_open" else close[t]
            costs[t], turnover[t] = book.trade_to(pending, px, dates[t])
        values[t] = book.value(close[t])
        nav[t] = book.cash + values[t].sum()
        if nav[t] <= 0:
            raise AccountWipedOutError(
                f"net asset value fell to zero or below on {dates[t]:%Y-%m-%d}"
            )

        view = market.up_to(dates[t])
        decision = align_targets(strategy.target_weights(view), view).iloc[-1].to_numpy(dtype=float)
        check_targets(decision, close[t], dates[t : t + 1], config.max_gross_leverage)
        pending = decision

    return BacktestResult(
        engine=ENGINE,
        strategy=strategy.name,
        fill=config.fill,
        nav=pd.Series(nav, index=dates, name="nav"),
        weights=pd.DataFrame(values / nav[:, None], index=dates, columns=market.assets),
        turnover=pd.Series(turnover, index=dates, name="turnover"),
        costs=pd.Series(costs, index=dates, name="costs"),
    )
