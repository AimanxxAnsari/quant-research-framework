"""Catch strategies that use the future.

A decision made at the close of date t must be the same whether or not the data after t
exists. The check recomputes the strategy on the market cut off at a sample of dates and
compares the last row with the full-history answer. Any difference means the strategy read
later data (a negative shift, a full-sample normalisation, a centred window, and so on).

The weights engine runs this automatically; the loop engine cannot look ahead by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.backtest.market import Market
from quant_research.backtest.strategy import Strategy
from quant_research.backtest.targets import align_targets
from quant_research.errors import LookAheadError


def check_no_lookahead(
    strategy: Strategy,
    market: Market,
    n_checks: int,
    full: pd.DataFrame | None = None,
    atol: float = 1e-12,
) -> None:
    """Raise ``LookAheadError`` if truncating the data changes any sampled past decision.

    The sample is ``n_checks`` dates spread evenly from the first date to the second-to-last
    (cutting at the last date removes nothing, so it cannot reveal anything).
    """
    dates = market.dates
    if n_checks <= 0 or len(dates) < 3:
        return
    reference = full if full is not None else align_targets(strategy.target_weights(market), market)
    rows = np.unique(np.linspace(0, len(dates) - 2, n_checks).round().astype(int))
    for i in rows:
        view = market.up_to(dates[i])
        decided = align_targets(strategy.target_weights(view), view).iloc[-1].to_numpy(dtype=float)
        expected = reference.iloc[i].to_numpy(dtype=float)
        if not np.allclose(decided, expected, rtol=0.0, atol=atol, equal_nan=True):
            gap = float(np.nanmax(np.abs(decided - expected)))
            raise LookAheadError(
                f"{strategy.name}: the decision on {dates[i]:%Y-%m-%d} changes by up to {gap:.3g} "
                "when later data is removed, so it uses information from after that date"
            )
