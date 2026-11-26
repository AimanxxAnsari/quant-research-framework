"""Cross-asset monitor: latest levels, changes and volatility context.

Writes a dated Markdown snapshot to ``data/snapshots/`` (local; contains current market data).

* Rates and spreads: change in basis points.
* Prices, FX, volatility indices: percentage change.
* Realised 21-day volatility and its percentile in the instrument's own history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.data import access
from quant_research.data.settings import Settings
from quant_research.reporting import md_table
from quant_research.stats import volatility as vol

RATES = {
    "DGS2": "UST 2Y",
    "DGS10": "UST 10Y",
    "T10Y2Y": "10Y-2Y",
    "DFII10": "10Y real",
    "T10YIE": "10Y breakeven",
}
LEVELS = {"VIXCLS": "VIX", "OVXCLS": "OVX"}
FX = {"DEXUSEU": "EUR/USD", "DEXUSUK": "GBP/USD", "DEXJPUS": "USD/JPY"}
ETFS = ["SPY", "QQQ", "EFA", "EEM", "TLT", "HYG", "GLD", "USO", "UUP"]
HORIZONS = {"1W": 5, "1M": 21, "3M": 63}


def _changes(s: pd.Series, bp: bool) -> dict[str, object]:
    s = s.dropna()
    last = s.iloc[-1]
    out = {"last": last, "date": f"{s.index[-1]:%Y-%m-%d}"}
    ytd_base = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
    bases = {k: s.iloc[-1 - n] for k, n in HORIZONS.items()}
    if len(ytd_base):
        bases["YTD"] = ytd_base.iloc[-1]
    for k, base in bases.items():
        out[k] = (last - base) * 100 if bp else (last / base - 1) * 100
    return out


def snapshot(settings: Settings) -> str:
    rates = access.series(settings, list(RATES))
    levels = access.series(settings, list(LEVELS))
    fx = access.series(settings, list(FX))
    px = access.prices(settings, ETFS)

    rate_tbl = pd.DataFrame({RATES[str(c)]: _changes(rates[c], bp=True) for c in rates}).T
    lvl_tbl = pd.DataFrame(
        {
            **{LEVELS[str(c)]: _changes(levels[c], False) for c in levels},
            **{FX[str(c)]: _changes(fx[c], False) for c in fx},
        }
    ).T
    etf_rows = {}
    for t in ETFS:
        row = _changes(px[t], bp=False)
        rv = vol.to_annual_vol(vol.close_to_close(px[t], 21)).dropna()
        row["21d vol %"] = rv.iloc[-1] * 100
        row["vol pctile"] = float((rv <= rv.iloc[-1]).mean() * 100)
        etf_rows[t] = row
    etf_tbl = pd.DataFrame(etf_rows).T
    for tbl in (rate_tbl, lvl_tbl, etf_tbl):
        tbl.index.name = ""
        for c in tbl.columns:
            if c not in ("date",):
                tbl[c] = tbl[c].map(lambda v: round(float(v), 2) if pd.notna(v) else np.nan)
    asof = max(rates.index.max(), px.index.max())
    return "\n".join(
        [
            f"# Cross-asset monitor ({asof:%Y-%m-%d})",
            "",
            "## Rates (changes in basis points)",
            "",
            md_table(rate_tbl, "{:.2f}"),
            "",
            "## Volatility indices and FX (changes in %)",
            "",
            md_table(lvl_tbl, "{:.2f}"),
            "",
            "## ETFs (changes in %, adjusted prices; realised vol and its historical percentile)",
            "",
            md_table(etf_tbl, "{:.2f}"),
            "",
        ]
    )


def write_snapshot(settings: Settings) -> str:
    text = snapshot(settings)
    out = settings.data_dir / "snapshots"
    out.mkdir(parents=True, exist_ok=True)
    date = text.split("(")[1].split(")")[0]
    (out / f"monitor_{date}.md").write_text(text)
    return text
