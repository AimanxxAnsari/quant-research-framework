"""Build a backtest ``Market`` from the warehouse: ETFs, FX spot and short rates (ADR-0009).

Calendar: the ETF trading days when any ETF is included, otherwise the FX fixing days. FX
fixings (FRED H.10, noon New York) are moved onto that calendar; on the few days the Fed is
closed but the exchange is open, the previous fixing is used, for at most ``FX_FILL_DAYS``
trading days. Short rates come from ``data.rates`` on calendar days and are read off on each
market date.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quant_research.backtest.market import Market
from quant_research.data import access, rates
from quant_research.data.settings import Settings
from quant_research.errors import ConfigError

FX_FILL_DAYS = 3


def fx_series_ids(settings: Settings) -> dict[str, str]:
    """Currency code -> FRED H.10 series id, from the instrument master."""
    fred = settings.instruments()["fred"]
    return {str(m["currency"]): sid for sid, m in fred.items() if m.get("currency")}


def _window(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    return frame.loc[start:end] if (start or end) else frame


def load_market(
    settings: Settings,
    etfs: Sequence[str] = (),
    fx: Sequence[str] = (),
    start: str | None = None,
    end: str | None = None,
) -> Market:
    """Market for the given ETF tickers and FX currency codes (e.g. ``["EUR", "JPY"]``)."""
    if not etfs and not fx:
        raise ConfigError("a market needs at least one ETF or currency")
    ids = fx_series_ids(settings)
    unknown = sorted(set(fx) - set(ids))
    if unknown:
        raise ConfigError(f"no FX series configured for {unknown}")

    fields = ("adj_open", "adj_high", "adj_low", "adj_close")
    frames: dict[str, list[pd.DataFrame]] = {f: [] for f in fields}
    if etfs:
        for f in fields:
            frames[f].append(access.prices(settings, list(etfs), field=f))
    calendar = frames["adj_close"][0].index if etfs else None
    if fx:
        spot = access.fx_usd_per_foreign(settings, [ids[c] for c in fx])
        spot.columns = list(fx)
        if calendar is not None:
            spot = spot.reindex(spot.index.union(calendar)).ffill(limit=FX_FILL_DAYS)
            spot = spot.reindex(calendar)
        for f in fields:  # one fixing a day: open = high = low = close
            frames[f].append(spot)
    close = _window(pd.concat(frames["adj_close"], axis=1), start, end).dropna(how="all")
    opens, high, low = (
        _window(pd.concat(frames[f], axis=1), start, end).reindex(close.index)
        for f in ("adj_open", "adj_high", "adj_low")
    )

    currencies = ["USD", *fx]
    # raises MissingValuesError, naming the currency, if a rate does not cover the period
    short = rates.short_rates(
        settings, currencies, start=close.index[0].date(), end=close.index[-1].date()
    ).reindex(close.index)
    asset_rate = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for c in fx:
        asset_rate[c] = short[c]
    return Market(
        open=opens,
        close=close,
        asset_class={**{t: "etf" for t in etfs}, **{c: "fx" for c in fx}},
        asset_rate=asset_rate,
        asset_day_count={
            **{t: 360 for t in etfs},
            **{c: rates.day_count(settings, c) for c in fx},
        },
        cash_rate=short["USD"],
        cash_day_count=rates.day_count(settings, "USD"),
        high_prices=high,
        low_prices=low,
    )
