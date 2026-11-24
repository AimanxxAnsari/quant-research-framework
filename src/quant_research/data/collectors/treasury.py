"""US Treasury daily par yield curves (nominal and real), one CSV per calendar year.

Tenor columns ("1 Mo", "1.5 Month", "2 Yr", ...) become series ids such as
``UST_PAR_1M``, ``UST_PAR_1.5M``, ``UST_PAR_10Y`` (nominal) and ``UST_REAL_10Y`` (real).
"""

from __future__ import annotations

import io
import re
from datetime import date

import pandas as pd

from quant_research.data.http import get
from quant_research.data.raw_store import RawStore, Snapshot
from quant_research.data.settings import Settings

SOURCE = "treasury"
URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
)
CURVES = {"nominal": "daily_treasury_yield_curve", "real": "daily_treasury_real_yield_curve"}
PREFIX = {"nominal": "UST_PAR", "real": "UST_REAL"}


def fetch(store: RawStore, settings: Settings, years: list[int] | None = None) -> list[Snapshot]:
    cfg = settings.instruments()["treasury"]
    this_year = date.today().year
    snaps = []
    for curve, type_ in CURVES.items():
        first = cfg["nominal_from_year"] if curve == "nominal" else cfg["real_from_year"]
        for year in years or range(first, this_year + 1):
            params = {"type": type_, "field_tdr_date_value": year, "page": "", "_format": "csv"}
            r = get(URL.format(year=year), params=params)
            snaps.append(store.save(SOURCE, f"{curve}_{year}", r.content, url=r.url, ext="csv"))
    return snaps


def tenor_code(column: str) -> str | None:
    """'1 Mo' -> '1M', '1.5 Month' -> '1.5M', '10 Yr' -> '10Y'; other columns -> None."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(mo|month|months|yr|year|years)\s*", column, re.I)
    if not m:
        return None
    return m.group(1) + ("M" if m.group(2).lower().startswith("mo") else "Y")


def parse(payload: bytes, dataset: str) -> pd.DataFrame:
    curve = dataset.split("_")[0]
    raw = pd.read_csv(io.BytesIO(payload))
    if raw.empty:
        return pd.DataFrame(columns=["series_id", "date", "value"])
    tenors = {c: tenor_code(c) for c in raw.columns}
    tenors = {c: t for c, t in tenors.items() if t}
    long = raw.melt(id_vars=["Date"], value_vars=list(tenors), var_name="col", value_name="value")
    long = long.dropna(subset=["value"])
    return pd.DataFrame(
        {
            "series_id": PREFIX[curve] + "_" + long["col"].map(tenors),
            "date": pd.to_datetime(long["Date"], format="%m/%d/%Y"),
            "value": long["value"].astype(float),
        }
    ).reset_index(drop=True)
