"""Data-quality checks on the clean tables. Failures must be visible, never silent.

Each check returns a row: table, check, subject, status (PASS / WARN / FAIL), detail.

* coverage    - every configured series / ticker is present (FAIL if missing)
* uniqueness  - primary keys (series_id, date) and (ticker, date) are unique (FAIL)
* missing     - no missing values in value / close (FAIL)
* range       - values inside plausible bounds for their asset class (FAIL)
* ohlc        - low <= open, close <= high and prices > 0 (FAIL)
* jumps       - adjusted daily returns beyond +-40% (WARN: data error or real event)
* freshness   - last observation recent enough for the series frequency (WARN)
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant_research.data.settings import Settings

# Plausible bounds by asset class (value units as stored). WTI spot went negative in 2020.
RANGES = {
    "rates": (-5.0, 25.0),
    "volatility": (5.0, 500.0),  # OVX reached ~325 in April 2020
    "fx": (1e-4, 1e4),
    "commodity": (-50.0, 500.0),
    "factor": (-25.0, 25.0),
}
MACRO_RANGES = {"CPIAUCSL": (10.0, 1000.0), "UNRATE": (0.0, 30.0)}
STALE_DAYS = {"daily": 10, "monthly": 75}  # FRED H.10 FX is published weekly with a lag


def _row(table, check, subject, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    return {"table": table, "check": check, "subject": subject, "status": status, "detail": detail}


def check_observations(
    obs: pd.DataFrame, series: pd.DataFrame, settings: Settings, as_of: date
) -> list[dict]:
    rows = []
    expected = set(settings.instruments()["fred"])
    present = set(obs["series_id"])
    missing = sorted(expected - present)
    rows.append(
        _row(
            "observations",
            "coverage",
            "fred series",
            not missing,
            f"missing: {missing}" if missing else f"{len(expected)} series",
        )
    )
    for src, needed in {"treasury": "UST_PAR_10Y", "french": "FF_MKT_RF"}.items():
        rows.append(
            _row(
                "observations",
                "coverage",
                src,
                needed in present,
                "" if needed in present else f"{needed} missing",
            )
        )

    dup = obs.duplicated(["series_id", "date"]).sum()
    rows.append(
        _row("observations", "uniqueness", "(series_id, date)", dup == 0, f"{dup} duplicate keys")
    )
    nan = obs["value"].isna().sum()
    rows.append(_row("observations", "missing", "value", nan == 0, f"{nan} missing values"))

    meta = series.set_index("series_id")
    for sid, g in obs.groupby("series_id", sort=True):
        ac = meta.loc[sid, "asset_class"] if sid in meta.index else None
        lo, hi = MACRO_RANGES.get(sid) or RANGES.get(ac, (float("-inf"), float("inf")))
        bad = g[(g["value"] < lo) | (g["value"] > hi)]
        if len(bad):
            first = bad.iloc[0]
            rows.append(
                _row(
                    "observations",
                    "range",
                    sid,
                    False,
                    f"{len(bad)} values outside [{lo}, {hi}], e.g. "
                    f"{first['value']} on {first['date']:%Y-%m-%d}",
                )
            )
        freq = meta.loc[sid, "frequency"] if sid in meta.index else "daily"
        age = (pd.Timestamp(as_of) - g["date"].max()).days
        limit = STALE_DAYS.get(freq, 10)
        if age > limit:
            rows.append(
                _row(
                    "observations",
                    "freshness",
                    sid,
                    False,
                    f"last {g['date'].max():%Y-%m-%d} ({age} days old)",
                    warn=True,
                )
            )
    rows.append(_row("observations", "range", "all series", True, "checked per series"))
    return rows


def check_prices(prices: pd.DataFrame, tickers: list[str], as_of: date) -> list[dict]:
    rows = []
    present = set(prices["ticker"]) if len(prices) else set()
    missing = sorted(set(tickers) - present)
    rows.append(
        _row(
            "prices",
            "coverage",
            "tickers",
            not missing,
            f"missing: {missing}" if missing else f"{len(tickers)} tickers",
        )
    )
    if not len(prices):
        return rows
    dup = prices.duplicated(["ticker", "date"]).sum()
    rows.append(_row("prices", "uniqueness", "(ticker, date)", dup == 0, f"{dup} duplicates"))
    nan = prices["close"].isna().sum() + prices["adj_close"].isna().sum()
    rows.append(_row("prices", "missing", "close / adj_close", nan == 0, f"{nan} missing"))
    p = prices
    bad = (
        (p[["open", "high", "low", "close", "adj_close"]] <= 0).any(axis=1)
        | (p["low"] > p[["open", "close"]].min(axis=1) + 1e-9)
        | (p["high"] < p[["open", "close"]].max(axis=1) - 1e-9)
    )
    rows.append(
        _row(
            "prices",
            "ohlc",
            "all tickers",
            not bad.any(),
            f"{int(bad.sum())} inconsistent bars"
            + (
                f", e.g. {p.loc[bad, 'ticker'].iloc[0]} {p.loc[bad, 'date'].iloc[0]:%Y-%m-%d}"
                if bad.any()
                else ""
            ),
        )
    )
    for t, g in p.groupby("ticker", sort=True):
        ret = g["adj_close"].pct_change().abs()
        jumps = g.loc[ret > 0.40, "date"]
        if len(jumps):
            rows.append(
                _row(
                    "prices",
                    "jumps",
                    t,
                    False,
                    f"{len(jumps)} daily moves > 40%, first {jumps.iloc[0]:%Y-%m-%d}",
                    warn=True,
                )
            )
        age = (pd.Timestamp(as_of) - g["date"].max()).days
        if age > STALE_DAYS["daily"]:
            rows.append(
                _row(
                    "prices",
                    "freshness",
                    t,
                    False,
                    f"last {g['date'].max():%Y-%m-%d} ({age} days old)",
                    warn=True,
                )
            )
    return rows


def run_checks(settings: Settings, as_of: date | None = None) -> pd.DataFrame:
    from quant_research.data.collectors.tiingo import all_tickers

    as_of = as_of or date.today()
    clean = settings.clean_dir
    obs = pd.read_parquet(clean / "observations.parquet")
    prices = pd.read_parquet(clean / "prices.parquet")
    series = pd.read_parquet(clean / "series.parquet")
    rows = check_observations(obs, series, settings, as_of)
    rows += check_prices(prices, all_tickers(settings), as_of)
    return pd.DataFrame(rows)


def health_markdown(results: pd.DataFrame, settings: Settings) -> str:
    counts = results["status"].value_counts().to_dict()
    lines = [
        "# Data health",
        "",
        f"PASS {counts.get('PASS', 0)} | WARN {counts.get('WARN', 0)} | "
        f"FAIL {counts.get('FAIL', 0)}",
        "",
        "| Table | Check | Subject | Status | Detail |",
        "|---|---|---|---|---|",
    ]
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    for r in results.sort_values("status", key=lambda s: s.map(order), kind="stable").itertuples():
        lines.append(f"| {r.table} | {r.check} | {r.subject} | {r.status} | {r.detail} |")
    return "\n".join(lines) + "\n"
