"""Short-term interest rates per currency, for FX carry and the return on cash.

Each currency's rate is spliced from one or more FRED series (``short_rates`` in
``config/instruments.yaml``; reasons in ADR-0008) onto a calendar-day index:

* daily series are carried over weekends and holidays for at most ``DAILY_FILL_DAYS`` days;
* monthly series (OECD averages, dated the first of the month) apply to every day of their
  month, and the last one is carried for at most ``MONTHLY_FILL_DAYS`` days while the next
  month is unpublished.

Units: percent per annum, simple money-market rates. Interest on a notional N held from date
t to t' is N * rate_t / 100 * (t' - t in calendar days) / day_count.

These are the rates actually earned over each day, so they are right for accounting (carry
accrual). They are not point-in-time: a monthly average is only published weeks later, so a
*signal* built on these rates must lag them first.

Gaps are never filled silently: a gap longer than the fill limit raises ``MissingValuesError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from quant_research.data import access
from quant_research.data.settings import Settings
from quant_research.errors import ConfigError, MissingValuesError

DAILY_FILL_DAYS = 7  # longest gap in these series is 5 days (Easter); 7 leaves some room
MONTHLY_FILL_DAYS = 62  # the month itself plus one unpublished month
DAY_COUNTS = (360, 365)


@dataclass(frozen=True)
class RateSegment:
    """One FRED series, used up to and including ``until`` (open-ended if None)."""

    series: str
    frequency: str
    until: date | None = None


@dataclass(frozen=True)
class ShortRateSpec:
    currency: str
    day_count: int
    segments: tuple[RateSegment, ...]


def _parse_segment(currency: str, raw: Mapping[str, Any], fred: Mapping[str, Any]) -> RateSegment:
    sid = raw.get("series")
    if not isinstance(sid, str) or sid not in fred:
        raise ConfigError(f"{currency}: series {sid!r} is not in the fred instrument list")
    freq = fred[sid].get("frequency")
    if freq not in ("daily", "monthly"):
        raise ConfigError(f"{currency}: series {sid} has unsupported frequency {freq!r}")
    until = raw.get("until")
    if until is not None and not isinstance(until, date):
        raise ConfigError(f"{currency}: 'until' for {sid} must be a date (YYYY-MM-DD)")
    return RateSegment(series=sid, frequency=freq, until=until)


def parse_specs(instruments: Mapping[str, Any]) -> dict[str, ShortRateSpec]:
    """Validate the ``short_rates`` config and return one spec per currency."""
    fred = instruments.get("fred", {})
    specs = {}
    for ccy, raw in (instruments.get("short_rates") or {}).items():
        day_count = raw.get("day_count")
        if day_count not in DAY_COUNTS:
            raise ConfigError(f"{ccy}: day_count must be one of {DAY_COUNTS}, got {day_count!r}")
        segs = tuple(_parse_segment(ccy, s, fred) for s in raw.get("segments") or [])
        if not segs:
            raise ConfigError(f"{ccy}: at least one segment is required")
        if segs[-1].until is not None or any(s.until is None for s in segs[:-1]):
            raise ConfigError(f"{ccy}: only the last segment may (and must) be open-ended")
        cutoffs = [s.until for s in segs[:-1] if s.until is not None]
        if cutoffs != sorted(set(cutoffs)):
            raise ConfigError(f"{ccy}: segment 'until' dates must be strictly increasing")
        specs[ccy] = ShortRateSpec(currency=ccy, day_count=day_count, segments=segs)
    return specs


def _to_calendar(obs: pd.Series, frequency: str, end: pd.Timestamp) -> pd.Series:
    """Observations on a calendar-day index, carried forward up to the fill limit."""
    obs = obs.dropna().sort_index()
    if obs.empty:
        return pd.Series(dtype=float)
    limit = DAILY_FILL_DAYS if frequency == "daily" else MONTHLY_FILL_DAYS
    days = pd.date_range(obs.index[0], max(end, obs.index[-1]), freq="D")
    return obs.reindex(days).ffill(limit=limit)


def splice(
    spec: ShortRateSpec,
    observations: Mapping[str, pd.Series],
    end: date,
    start: date | None = None,
) -> pd.Series:
    """One calendar-day series for a currency, from its segments' raw observations.

    The result runs from ``start`` (default: the first observation of the first segment) to
    ``end``; segments that end before ``start`` are skipped. Raises ``MissingValuesError`` if
    any day in that range has no rate (a gap longer than the fill limit, a segment with no
    data, or data that starts after ``start`` or stops before ``end``).
    """
    stop = pd.Timestamp(end)
    begin = pd.Timestamp(start) if start is not None else None
    pieces = []
    lo: pd.Timestamp | None = begin
    for seg in spec.segments:
        hi = pd.Timestamp(seg.until) if seg.until is not None else stop
        if begin is not None and hi < begin:
            continue  # this segment is over before the period we need
        cal = _to_calendar(
            observations.get(seg.series, pd.Series(dtype=float)), seg.frequency, stop
        )
        piece = cal.loc[lo:hi] if lo is not None else cal.loc[:hi]
        if piece.empty:
            raise MissingValuesError(f"{spec.currency}: no {seg.series} data up to {hi:%Y-%m-%d}")
        pieces.append(piece)
        lo = hi + pd.Timedelta(days=1)

    out = pd.concat(pieces)
    full = pd.date_range(begin if begin is not None else out.index[0], stop, freq="D")
    out = out.reindex(full).rename(spec.currency)
    missing = out[out.isna()]
    if len(missing):
        first = f"{missing.index[0]:%Y-%m-%d}"
        raise MissingValuesError(
            f"{spec.currency}: no short rate on {len(missing)} days, first {first}"
            " (gap longer than the fill limit, or data does not cover the requested period)"
        )
    return out


def short_rates(
    settings: Settings,
    currencies: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Calendar-day short rates (percent per annum), one column per currency.

    Without ``start`` each column begins where its own history begins (earlier rows are NaN);
    with it, every column must be complete from ``start``. Columns run without gaps to ``end``
    (default: the latest observation across the series used).
    """
    specs = parse_specs(settings.instruments())
    wanted = currencies or list(specs)
    unknown = sorted(set(wanted) - set(specs))
    if unknown:
        raise ConfigError(f"no short_rates config for {unknown}")
    ids = sorted({s.series for c in wanted for s in specs[c].segments})
    raw = access.series(settings, ids)
    obs = {sid: raw[sid].dropna() for sid in ids}
    stop = end or max(s.index.max() for s in obs.values()).date()
    cols = [splice(specs[c], obs, stop, start) for c in wanted]
    return pd.concat(cols, axis=1)


def day_count(settings: Settings, currency: str) -> int:
    """Money-market day-count basis (360 or 365) for the currency."""
    specs = parse_specs(settings.instruments())
    if currency not in specs:
        raise ConfigError(f"no short_rates config for {currency}")
    return specs[currency].day_count
