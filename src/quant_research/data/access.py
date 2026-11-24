"""Read clean tables into analysis-ready (wide) DataFrames.

All functions return DataFrames indexed by date (tz-naive, daily), one column per series or
ticker, sorted, with no forward-filling: missing days stay missing so that callers decide
explicitly how to align series with different calendars.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from quant_research.data.settings import Settings


def _con(settings: Settings) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(settings.warehouse), read_only=True)


def series(settings: Settings, ids: list[str], start: str | None = None) -> pd.DataFrame:
    """Wide table of observation values for the given series ids."""
    q = "SELECT series_id, date, value FROM observations WHERE series_id IN ?"
    params: list = [ids]
    if start:
        q += " AND date >= ?"
        params.append(start)
    with _con(settings) as con:
        long = con.execute(q, params).df()
    wide = long.pivot(index="date", columns="series_id", values="value").sort_index()
    missing = sorted(set(ids) - set(wide.columns))
    if missing:
        raise KeyError(f"series not in warehouse: {missing}")
    return wide[ids]


def prices(
    settings: Settings, tickers: list[str], field: str = "adj_close", start: str | None = None
) -> pd.DataFrame:
    """Wide table of one price field (default adjusted close) for the given tickers."""
    allowed = {"open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"}
    if field not in allowed:
        raise ValueError(f"field must be one of {sorted(allowed)}")
    q = f"SELECT ticker, date, {field} AS v FROM prices WHERE ticker IN ?"
    params: list = [tickers]
    if start:
        q += " AND date >= ?"
        params.append(start)
    with _con(settings) as con:
        long = con.execute(q, params).df()
    wide = long.pivot(index="date", columns="ticker", values="v").sort_index()
    missing = sorted(set(tickers) - set(wide.columns))
    if missing:
        raise KeyError(f"tickers not in warehouse: {missing}")
    return wide[tickers]


def ohlc(settings: Settings, ticker: str, adjusted: bool = True) -> pd.DataFrame:
    """Open/high/low/close for one ticker (adjusted by default, so overnight gaps across
    dividends and splits are not mistaken for returns)."""
    p = "adj_" if adjusted else ""
    q = (
        f"SELECT date, {p}open AS open, {p}high AS high, {p}low AS low, {p}close AS close "
        "FROM prices WHERE ticker = ? ORDER BY date"
    )
    with _con(settings) as con:
        df = con.execute(q, [ticker]).df()
    if df.empty:
        raise KeyError(f"ticker not in warehouse: {ticker}")
    return df.set_index("date")


def fx_usd_per_foreign(settings: Settings, ids: list[str]) -> pd.DataFrame:
    """FX rates expressed uniformly as USD per unit of foreign currency.

    FRED H.10 series come in two conventions (e.g. DEXUSEU = USD per EUR, DEXJPUS = JPY per
    USD); the instrument master records which, and 'foreign per USD' series are inverted.
    A rise then always means the foreign currency strengthened against the dollar.
    """
    meta = settings.instruments()["fred"]
    wide = series(settings, ids)
    for sid in ids:
        quote = meta[sid].get("quote")
        if quote == "foreign_per_USD":
            wide[sid] = 1.0 / wide[sid]
        elif quote != "USD_per_foreign":
            raise ValueError(f"{sid} is not an FX series with a known quote convention")
    return wide
