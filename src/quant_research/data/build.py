"""Build clean tables from the latest raw snapshot of every dataset.

Outputs (``data/clean/``), all sorted deterministically:

* ``observations.parquet`` - long table: series_id, date, value, source_id, retrieved_at,
  raw_sha256 (FRED, Treasury, French).
* ``prices.parquet`` - ETF end-of-day prices, raw and adjusted (Tiingo).
* ``series.parquet`` - catalogue: series_id, source_id, name, asset_class, unit, frequency.
* ``warehouse.duckdb`` - DuckDB views over the Parquet files for SQL access.

The build reads only the raw layer, so running it twice on the same raw data gives
identical tables (tested).
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

from quant_research.data.collectors import COLLECTORS
from quant_research.data.raw_store import RawStore
from quant_research.data.settings import Settings

OBS_SOURCES = ("fred", "treasury", "french")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)  # atomic: a failed build never leaves a half-written table


def _with_lineage(df: pd.DataFrame, source: str, snap) -> pd.DataFrame:
    return df.assign(source_id=source, retrieved_at=snap.retrieved_at, raw_sha256=snap.sha256)


def build_observations(store: RawStore) -> pd.DataFrame:
    frames = []
    for source in OBS_SOURCES:
        for snap in store.latest_all(source):
            df = COLLECTORS[source].parse(store.read(snap), snap.dataset)
            frames.append(_with_lineage(df, source, snap))
    if not frames:
        return pd.DataFrame(
            columns=["series_id", "date", "value", "source_id", "retrieved_at", "raw_sha256"]
        )
    obs = pd.concat(frames, ignore_index=True)
    return obs.sort_values(["series_id", "date"], kind="stable").reset_index(drop=True)


def build_prices(store: RawStore) -> pd.DataFrame:
    frames = [
        _with_lineage(COLLECTORS["tiingo"].parse(store.read(s), s.dataset), "tiingo", s)
        for s in store.latest_all("tiingo")
    ]
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close"])
    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


def build_series(obs: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    fred = settings.instruments()["fred"]
    rows = []
    for sid, src in obs[["series_id", "source_id"]].drop_duplicates().itertuples(index=False):
        if src == "fred":
            meta = fred.get(sid, {})
            rows.append({"series_id": sid, "source_id": src, "name": meta.get("name", sid),
                         "asset_class": meta.get("asset_class"), "unit": meta.get("unit"),
                         "frequency": meta.get("frequency")})
        elif src == "treasury":
            kind = "real" if sid.startswith("UST_REAL") else "par"
            rows.append({"series_id": sid, "source_id": src,
                         "name": f"UST {kind} yield {sid.rsplit('_', 1)[1]}",
                         "asset_class": "rates", "unit": "percent", "frequency": "daily"})
        else:
            rows.append({"series_id": sid, "source_id": src, "name": f"Fama-French {sid[3:]}",
                         "asset_class": "factor", "unit": "percent per day",
                         "frequency": "daily"})
    cols = ["series_id", "source_id", "name", "asset_class", "unit", "frequency"]
    return pd.DataFrame(rows, columns=cols).sort_values("series_id").reset_index(drop=True)


def build(settings: Settings) -> dict[str, int]:
    store = RawStore(settings.raw_dir)
    obs = build_observations(store)
    prices = build_prices(store)
    series = build_series(obs, settings)
    clean = settings.clean_dir
    _write_parquet(obs, clean / "observations.parquet")
    _write_parquet(prices, clean / "prices.parquet")
    _write_parquet(series, clean / "series.parquet")

    with duckdb.connect(str(settings.warehouse)) as con:
        for name in ("observations", "prices", "series"):
            path = (clean / f"{name}.parquet").as_posix()
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    return {"observations": len(obs), "prices": len(prices), "series": len(series)}
