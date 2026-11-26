"""Strategies turn market data into target portfolio weights.

A strategy returns, for every date in the market it is given, the weights it wants to hold
after seeing that date's close: one column per asset, as a fraction of net asset value
(negative = short, the rest is USD cash, which may be negative = borrowing). The engine fills
them later, at the next open or next close.

The contract that matters most: the row for date t may only use data up to and including the
close of date t. ``lookahead.check_no_lookahead`` tests this by recomputing on truncated data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from quant_research.backtest.market import Market
from quant_research.errors import InvalidParameterError, MisalignedDataError


class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    def target_weights(self, market: Market) -> pd.DataFrame:
        """Target weights decided at each date's close, indexed like ``market.dates``."""
        ...


@dataclass(frozen=True)
class FixedWeights:
    """Hold constant weights, rebalanced every day; flat until every asset has a price."""

    weights: Mapping[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        parts = ", ".join(f"{a} {w:g}" for a, w in self.weights.items())
        return f"fixed weights ({parts})" if parts else "all cash"

    def target_weights(self, market: Market) -> pd.DataFrame:
        unknown = sorted(set(self.weights) - set(market.assets))
        if unknown:
            raise MisalignedDataError(f"weights for assets not in the market: {unknown}")
        out = pd.DataFrame(0.0, index=market.dates, columns=market.assets)
        if not self.weights:
            return out
        cols = list(self.weights)
        listed = market.close[cols].notna().all(axis=1)
        for a, w in self.weights.items():
            out.loc[listed, a] = w
        return out


@dataclass(frozen=True)
class MovingAverageCrossover:
    """Long one asset when its fast moving average of closes is above the slow one.

    Below it the position is flat, or short if ``allow_short``. Averages use closes up to and
    including the decision date; until the slow average exists the position is flat.
    """

    asset: str
    fast: int
    slow: int
    allow_short: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.fast < self.slow:
            raise InvalidParameterError("need 1 <= fast < slow")

    @property
    def name(self) -> str:
        side = "long/short" if self.allow_short else "long/flat"
        return f"MA {self.fast}/{self.slow} {side} on {self.asset}"

    def target_weights(self, market: Market) -> pd.DataFrame:
        if self.asset not in market.assets:
            raise MisalignedDataError(f"{self.asset} is not in the market")
        close = market.close[self.asset]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        below = -1.0 if self.allow_short else 0.0
        signal = (fast > slow).astype(float).where(fast > slow, below)
        signal = signal.where(slow.notna(), 0.0)
        out = pd.DataFrame(0.0, index=market.dates, columns=market.assets)
        out[self.asset] = signal
        return out
