"""Tiingo end-of-day prices (raw and split/dividend-adjusted), one JSON payload per ticker.

Free tier: about 50 requests per hour. Licence: internal use only; never redistribute.
The API token is sent in the Authorization header, never in the URL.
"""

from __future__ import annotations

import json

import pandas as pd

from quant_research.data.http import get
from quant_research.data.raw_store import RawStore, Snapshot
from quant_research.data.settings import Settings, secret

SOURCE = "tiingo"
URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
COLUMNS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "adjOpen": "adj_open",
    "adjHigh": "adj_high",
    "adjLow": "adj_low",
    "adjClose": "adj_close",
    "adjVolume": "adj_volume",
    "divCash": "div_cash",
    "splitFactor": "split_factor",
}


def all_tickers(settings: Settings) -> list[str]:
    groups = settings.instruments()["tiingo"]["etfs"]
    return [t for group in groups.values() for t in group]


def fetch(store: RawStore, settings: Settings, tickers: list[str] | None = None) -> list[Snapshot]:
    start = settings.instruments()["tiingo"]["start_date"]
    headers = {"Authorization": f"Token {secret('TIINGO_API_KEY')}"}
    snaps = []
    for t in tickers or all_tickers(settings):
        r = get(URL.format(ticker=t.lower()), params={"startDate": start}, headers=headers)
        snaps.append(store.save(SOURCE, t.upper(), r.content, url=r.url, ext="json"))
    return snaps


def parse(payload: bytes, dataset: str) -> pd.DataFrame:
    data = json.loads(payload)
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date", *COLUMNS.values()])
    out = df.rename(columns=COLUMNS)[list(COLUMNS.values())].astype(float)
    out.insert(0, "date", pd.to_datetime(df["date"].str[:10], format="%Y-%m-%d"))
    out.insert(0, "ticker", dataset)
    return out.reset_index(drop=True)
