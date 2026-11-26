"""Synthetic markets and strategies for backtest tests. No vendor data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research.backtest.market import Market


def dates(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def make_market(
    close: pd.DataFrame,
    opens: pd.DataFrame | None = None,
    fx: Sequence[str] = (),
    asset_rate: Mapping[str, float | pd.Series] | None = None,
    cash_rate: float | pd.Series = 0.0,
    day_counts: Mapping[str, int] | None = None,
) -> Market:
    """Market from closes (and opens; default = closes). Columns in ``fx`` are currencies."""
    idx = close.index
    rate = pd.DataFrame(0.0, index=idx, columns=close.columns)
    for a, r in (asset_rate or {}).items():
        rate[a] = r
    cash = cash_rate if isinstance(cash_rate, pd.Series) else pd.Series(cash_rate, index=idx)
    return Market(
        open=opens if opens is not None else close.copy(),
        close=close,
        asset_class={a: ("fx" if a in fx else "etf") for a in close.columns},
        asset_rate=rate,
        asset_day_count={a: (day_counts or {}).get(a, 360) for a in close.columns},
        cash_rate=cash,
    )


def random_market(seed: int, n: int = 260, n_etf: int = 3, n_fx: int = 2) -> Market:
    """GBM-like ETFs with overnight gaps (one listed late), FX with moving short rates."""
    rng = np.random.default_rng(seed)
    idx = dates(n)
    etfs = [f"E{i}" for i in range(n_etf)]
    fx = [f"F{i}" for i in range(n_fx)]
    close, opens = {}, {}
    for a in etfs:
        night = rng.normal(0, 0.006, n)
        day = rng.normal(0.0003, 0.01, n)
        log_open = np.log(50 + 100 * rng.random()) + np.cumsum(night + np.r_[0.0, day[:-1]])
        opens[a] = np.exp(log_open)
        close[a] = np.exp(log_open + day)
    for a in fx:
        close[a] = opens[a] = np.exp(np.cumsum(rng.normal(0, 0.005, n))) * (0.5 + rng.random())
    c, o = pd.DataFrame(close, index=idx), pd.DataFrame(opens, index=idx)
    late = etfs[-1]
    c.loc[c.index[: n // 4], late] = np.nan
    o.loc[o.index[: n // 4], late] = np.nan
    rates = {a: pd.Series(2 + np.cumsum(rng.normal(0, 0.05, n)), index=idx) for a in fx}
    cash = pd.Series(3 + np.cumsum(rng.normal(0, 0.05, n)), index=idx)
    return make_market(
        c, o, fx=fx, asset_rate=rates, cash_rate=cash, day_counts={fx[0]: 365} if fx else None
    )


@dataclass(frozen=True)
class RandomWeights:
    """Random targets that change on about a third of days, drawn per date from a seed.

    Each date's draw depends only on the seed and that date, and the forward fill only looks
    backwards, so cutting off later data never changes an earlier decision (no look-ahead).
    """

    seed: int
    max_abs: float = 0.8

    @property
    def name(self) -> str:
        return f"random weights {self.seed}"

    def target_weights(self, market: Market) -> pd.DataFrame:
        rows = []
        for d in market.dates:
            rng = np.random.default_rng([self.seed, d.toordinal()])
            change = rng.random() < 0.35
            w = rng.uniform(-self.max_abs, self.max_abs, len(market.assets))
            rows.append(w if change else np.full(len(market.assets), np.nan))
        out = pd.DataFrame(rows, index=market.dates, columns=market.assets).ffill().fillna(0.0)
        return out.where(market.close.notna(), 0.0)


@dataclass(frozen=True)
class FromFrame:
    """Targets taken from a fixed table (only for hand-built known-answer tests)."""

    table: pd.DataFrame
    label: str = "from table"

    @property
    def name(self) -> str:
        return self.label

    def target_weights(self, market: Market) -> pd.DataFrame:
        return self.table.reindex(market.dates)


@dataclass(frozen=True)
class PeekTomorrow:
    """Planted look-ahead: long when the return the position will earn is positive.

    A decision at close t is filled at close t+1 and earns close t+1 -> t+2, so that is the
    future return this bug reads. (Peeking only at t -> t+1 would earn nothing: the fill lag
    already moves the position past it.)"""

    asset: str

    @property
    def name(self) -> str:
        return f"peek at tomorrow ({self.asset})"

    def target_weights(self, market: Market) -> pd.DataFrame:
        c = market.close[self.asset]
        out = pd.DataFrame(0.0, index=market.dates, columns=market.assets)
        out[self.asset] = (c.shift(-2) > c.shift(-1)).astype(float)
        return out
