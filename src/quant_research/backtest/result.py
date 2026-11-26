"""What a backtest returns. Both engines fill in the same fields, so they can be compared."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Daily outputs of one backtest run.

    nav: net asset value in USD at each close, after any trade made at that close.
    weights: value of each asset / nav at each close.
    turnover: traded notional / net asset value just before the trade, per date.
    costs: transaction costs paid, in USD, per date.
    """

    engine: str
    strategy: str
    fill: str
    nav: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series

    @property
    def returns(self) -> pd.Series:
        """Simple daily returns of the net asset value."""
        return self.nav.pct_change().iloc[1:]
