"""Kenneth French Data Library: daily factor files (zipped CSV with text preamble).

Values are daily returns in percent. Series ids: FF_MKT_RF, FF_SMB, FF_HML, FF_RF, FF_MOM.
"""

from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

from quant_research.data.http import get
from quant_research.data.raw_store import RawStore, Snapshot
from quant_research.data.settings import Settings
from quant_research.errors import DataIntegrityError

SOURCE = "french"
URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"


def fetch(store: RawStore, settings: Settings) -> list[Snapshot]:
    snaps = []
    for name in settings.instruments()["french"]["datasets"]:
        r = get(URL.format(name=name))
        snaps.append(store.save(SOURCE, name, r.content, url=r.url, ext="zip"))
    return snaps


def series_id(column: str) -> str:
    return "FF_" + re.sub(r"[^A-Z0-9]+", "_", column.strip().upper()).strip("_")


def parse(payload: bytes, dataset: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            # The header row starts with an empty cell followed by factor names.
            if len(cells) > 1 and cells[0] == "" and all(cells[1:]):
                header = cells[1:]
            continue
        if re.fullmatch(r"\d{8}", cells[0]):
            rows.append(cells)
        elif rows:  # first non-data line after the daily block ends it
            break
    if header is None or not rows:
        raise DataIntegrityError(f"could not find a daily data block in {dataset}")
    df = pd.DataFrame(rows, columns=["date", *header])
    long = df.melt(id_vars="date", var_name="col", value_name="value")
    return (
        pd.DataFrame(
            {
                "series_id": long["col"].map(series_id),
                "date": pd.to_datetime(long["date"], format="%Y%m%d"),
                "value": pd.to_numeric(long["value"], errors="coerce"),
            }
        )
        .dropna(subset=["value"])
        .reset_index(drop=True)
    )
