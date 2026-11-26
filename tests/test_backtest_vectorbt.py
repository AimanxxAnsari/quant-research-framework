"""Cross-check against an independent library: vectorbt (ADR-0010).

vectorbt is told to trade to the same target percentages at the same prices with the same
fees. It differs from our engines in one convention: it never lets USD cash go negative, so it
shrinks any order that would need borrowed cash (long weights above 100% of NAV, or exactly
100% plus fees). Our engines borrow at the cash rate instead. The tests therefore use targets
that never need negative cash; under that condition the NAV paths must match to rounding
error, with and without costs. (Found while writing these tests: weights of 119% and 29% long
were cut to 71% and 29% by vectorbt.)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from bt_helpers import RandomWeights, make_market, random_market

from quant_research.backtest.config import BacktestConfig, CostModel
from quant_research.backtest.market import Market
from quant_research.backtest.runner import run
from quant_research.backtest.strategy import MovingAverageCrossover, Strategy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import vectorbt as vbt


@dataclass(frozen=True)
class Scaled:
    inner: Strategy
    factor: float

    @property
    def name(self) -> str:
        return f"{self.inner.name} x{self.factor:g}"

    def target_weights(self, market: Market) -> pd.DataFrame:
        return self.inner.target_weights(market) * self.factor


def etf_market(seed: int) -> Market:
    full = random_market(seed, n_etf=2, n_fx=0)
    # vectorbt has no interest accrual or late listing, so keep both out of this comparison
    close = full.close.ffill().bfill()
    opens = full.open.ffill().bfill()
    return make_market(close, opens)


def vectorbt_nav(strategy: Strategy, market: Market, fill: str, fee: float) -> np.ndarray:
    size = strategy.target_weights(market).shift(1)  # decided at close t-1, traded at bar t
    pf = vbt.Portfolio.from_orders(
        close=market.close,
        size=size,
        size_type="targetpercent",
        price=market.open if fill == "next_open" else market.close,
        fees=fee,
        direction="both",
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
        init_cash=1e6,
        freq="1D",
    )
    return pf.value().to_numpy()


def ours(strategy: Strategy, market: Market, fill: str, bps: float) -> np.ndarray:
    config = BacktestConfig(fill=fill, initial_nav=1e6, costs=CostModel({"etf": bps, "fx": 0.0}))
    return run(strategy, market, config, "weights").nav.to_numpy()


STRATEGIES = [
    MovingAverageCrossover("E0", 5, 20, allow_short=True),
    RandomWeights(3, max_abs=0.5),  # long weights sum to at most 100%
]


@pytest.mark.parametrize("fill", ["next_open", "next_close"])
@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_matches_vectorbt_without_costs(strategy, fill):
    m = etf_market(11)
    np.testing.assert_allclose(
        ours(strategy, m, fill, 0.0), vectorbt_nav(strategy, m, fill, 0.0), rtol=1e-12
    )


@pytest.mark.parametrize("fill", ["next_open", "next_close"])
@pytest.mark.parametrize(
    "strategy",
    [Scaled(STRATEGIES[0], 0.9), RandomWeights(3, max_abs=0.45)],
    ids=lambda s: s.name,
)
def test_matches_vectorbt_with_costs_when_cash_covers_fees(strategy, fill):
    m = etf_market(11)
    np.testing.assert_allclose(
        ours(strategy, m, fill, 5.0), vectorbt_nav(strategy, m, fill, 5e-4), rtol=1e-12
    )
