"""FRED: one JSON payload per series (full history), ``/fred/series/observations``.

Missing values are encoded as "." by FRED (e.g. market holidays) and are dropped.
"""

from __future__ import annotations

import json

import pandas as pd

from quant_research.data.http import get
from quant_research.data.raw_store import RawStore, Snapshot
from quant_research.data.settings import Settings, secret

SOURCE = "fred"
URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch(store: RawStore, settings: Settings, series: list[str] | None = None) -> list[Snapshot]:
    series = series or list(settings.instruments()["fred"])
    key = secret("FRED_API_KEY")
    snaps = []
    for sid in series:
        r = get(URL, params={"series_id": sid, "api_key": key, "file_type": "json"})
        snaps.append(store.save(SOURCE, sid, r.content, url=r.url, ext="json"))
    return snaps


def parse(payload: bytes, dataset: str) -> pd.DataFrame:
    obs = json.loads(payload)["observations"]
    df = pd.DataFrame(obs, columns=["date", "value"])
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    df = df.dropna(subset=["value"])
    return pd.DataFrame(
        {
            "series_id": dataset,
            "date": pd.to_datetime(df["date"], format="%Y-%m-%d"),
            "value": df["value"].astype(float),
        }
    ).reset_index(drop=True)
