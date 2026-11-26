"""The target-weight engine: fast, works in return space (ADR-0010).

The strategy is asked once for its whole history of target weights. The engine then follows
net asset value (NAV) through time using weights and growth factors only:

Let W_t be the weights filled at trade point tau_t (the open or close of day t; W_0 = 0) and
held until tau_{t+1}; q_t the growth of each asset's value from tau_{t-1} to tau_t (price
ratio times interest on FX balances); g_t the growth of USD cash from interest; k the cost per
unit of traded notional. With X_t the NAV just before the trade at tau_t:

    X_t / X_{t-1} = rho_t = sum_i W_{t-1,i} q_{t,i} + (1 - sum_i W_{t-1,i}) g_t - kappa_{t-1} g_t

where kappa_t = sum_i k_i |W_{t,i} - W_{t-1,i} q_{t,i} / rho_t| is the cost of the trade at tau_t
as a fraction of X_t (the second term is the weight each asset has drifted to by tau_t). With
no costs NAV is a plain cumulative product; with costs rho and kappa form a scalar recursion
over time, while every step stays vectorised across assets.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quant_research.backtest.config import BacktestConfig
from quant_research.backtest.lookahead import check_no_lookahead
from quant_research.backtest.market import Market
from quant_research.backtest.result import BacktestResult
from quant_research.backtest.strategy import Strategy
from quant_research.backtest.targets import align_targets, check_targets
from quant_research.errors import AccountWipedOutError, MissingValuesError

ENGINE = "weights"


def _segment_growth(prices: np.ndarray, asset_accrual: np.ndarray, held: np.ndarray) -> np.ndarray:
    """q_t: growth of a unit of value in each asset from one trade point to the next."""
    q = np.ones_like(prices)
    with np.errstate(invalid="ignore", divide="ignore"):
        q[1:] = prices[1:] / prices[:-1] * asset_accrual[1:]
    unpriced = np.isnan(q)
    if (unpriced[1:] & (held[:-1] != 0)).any():
        raise MissingValuesError("a held asset has no price at the next trade point")
    return np.where(unpriced, 1.0, q)


def _nav_path(
    w: np.ndarray, q: np.ndarray, g: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rho (NAV growth between trade points), kappa (cost / NAV) and turnover per date."""
    held = w.sum(axis=1)
    a = np.ones(len(g))
    a[1:] = (w[:-1] * q[1:]).sum(axis=1) + (1.0 - held[:-1]) * g[1:]
    rho, kappa, turnover = a.copy(), np.zeros(len(g)), np.zeros(len(g))
    if not k.any():
        drift = np.zeros_like(w)
        drift[1:] = w[:-1] * q[1:] / rho[1:, None]
        turnover = np.abs(w - drift).sum(axis=1)
        return rho, kappa, turnover
    for t in range(1, len(g)):
        rho[t] = a[t] - kappa[t - 1] * g[t]
        trade = np.abs(w[t] - w[t - 1] * q[t] / rho[t])
        turnover[t] = trade.sum()
        kappa[t] = (trade * k).sum()
    return rho, kappa, turnover


def simulate(targets: pd.DataFrame, market: Market, config: BacktestConfig) -> BacktestResult:
    """Run already-checked target weights (decided at each close) through the market."""
    d = targets.to_numpy(dtype=float)
    w = np.zeros_like(d)
    w[1:] = d[:-1]  # decided at close t-1, filled at the next trade point
    close = market.close.to_numpy(dtype=float)
    opens = market.open.to_numpy(dtype=float)
    trade_px = opens if config.fill == "next_open" else close
    g, asset_accrual = market.accrual_factors()
    q = _segment_growth(trade_px, asset_accrual, w)
    k = config.costs.fractions([market.asset_class[a] for a in market.assets])

    rho, kappa, turnover = _nav_path(w, q, g, k)
    if (rho <= 0).any():
        bad = market.dates[int(np.argmax(rho <= 0))]
        raise AccountWipedOutError(f"net asset value fell to zero or below on {bad:%Y-%m-%d}")
    x = config.initial_nav * np.cumprod(rho)  # NAV just before each trade

    if config.fill == "next_open":
        with np.errstate(invalid="ignore", divide="ignore"):
            intraday = np.where(w != 0, close / opens, 1.0)
        values = w * intraday * x[:, None]
        nav = x * ((w * intraday).sum(axis=1) + 1.0 - w.sum(axis=1) - kappa)
    else:
        values = w * x[:, None]
        nav = x * (1.0 - kappa)
    if (nav <= 0).any():
        bad = market.dates[int(np.argmax(nav <= 0))]
        raise AccountWipedOutError(f"net asset value fell to zero or below on {bad:%Y-%m-%d}")

    idx = market.dates
    return BacktestResult(
        engine=ENGINE,
        strategy="",
        fill=config.fill,
        nav=pd.Series(nav, index=idx, name="nav"),
        weights=pd.DataFrame(values / nav[:, None], index=idx, columns=market.assets),
        turnover=pd.Series(turnover, index=idx, name="turnover"),
        costs=pd.Series(kappa * x, index=idx, name="costs"),
    )


def run(strategy: Strategy, market: Market, config: BacktestConfig) -> BacktestResult:
    """Ask the strategy for all its target weights at once, check them, and simulate."""
    targets = align_targets(strategy.target_weights(market), market)
    check_targets(
        targets.to_numpy(dtype=float),
        market.close.to_numpy(dtype=float),
        market.dates,
        config.max_gross_leverage,
    )
    if config.lookahead_checks:
        check_no_lookahead(strategy, market, config.lookahead_checks, full=targets)
    return replace(simulate(targets, market, config), strategy=strategy.name)
