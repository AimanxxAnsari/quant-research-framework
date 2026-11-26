"""Short-rate splicing for FX carry (data/rates.py), on synthetic observations."""

from datetime import date

import pandas as pd
import pytest

from quant_research.data.quality import check_short_rates
from quant_research.data.rates import (
    DAILY_FILL_DAYS,
    MONTHLY_FILL_DAYS,
    RateSegment,
    ShortRateSpec,
    parse_specs,
    splice,
)
from quant_research.errors import ConfigError, MissingValuesError


def _daily(start: str, end: str, value: float) -> pd.Series:
    return pd.Series(value, index=pd.bdate_range(start, end))  # weekdays only, like FRED


def _spec(*segments: RateSegment, ccy: str = "EUR") -> ShortRateSpec:
    return ShortRateSpec(currency=ccy, day_count=360, segments=segments)


def test_segments_switch_exactly_after_until():
    spec = _spec(
        RateSegment("OLD", "daily", until=date(2019, 9, 30)),
        RateSegment("NEW", "daily"),
    )
    obs = {
        "OLD": _daily("2019-09-02", "2019-10-31", 1.0),  # overlaps NEW on purpose
        "NEW": _daily("2019-10-01", "2019-10-31", 2.0),
    }
    out = splice(spec, obs, date(2019, 10, 31))
    assert out[pd.Timestamp("2019-09-30")] == 1.0
    assert out[pd.Timestamp("2019-10-01")] == 2.0
    assert out.index[0] == pd.Timestamp("2019-09-02")
    assert out.index.freq == "D" and not out.isna().any()


def test_weekends_are_carried_but_long_daily_gaps_fail():
    spec = _spec(RateSegment("X", "daily"))
    obs = _daily("2024-01-01", "2024-03-29", 5.0)
    out = splice(spec, {"X": obs}, date(2024, 3, 31))
    assert out[pd.Timestamp("2024-01-06")] == 5.0  # Saturday carries Friday's rate
    holed = obs.drop(obs.loc["2024-02-01":"2024-02-20"].index)
    with pytest.raises(MissingValuesError, match="EUR"):
        splice(spec, {"X": holed}, date(2024, 3, 31))


def test_monthly_value_covers_its_month_and_one_unpublished_month():
    spec = _spec(RateSegment("M", "monthly"), ccy="JPY")
    obs = pd.Series([0.1, 0.2], index=pd.to_datetime(["2026-09-01", "2026-10-01"]))
    out = splice(spec, {"M": obs}, date(2026, 11, 26))
    assert out[pd.Timestamp("2026-09-30")] == 0.1
    assert out[pd.Timestamp("2026-10-31")] == 0.2
    assert out[pd.Timestamp("2026-11-26")] == 0.2  # November not yet published
    too_late = (pd.Timestamp("2026-10-01") + pd.Timedelta(days=MONTHLY_FILL_DAYS + 1)).date()
    with pytest.raises(MissingValuesError, match="JPY"):
        splice(spec, {"M": obs}, too_late)


def test_missing_segment_data_fails_loudly():
    spec = _spec(
        RateSegment("OLD", "daily", until=date(2019, 9, 30)),
        RateSegment("NEW", "daily"),
    )
    with pytest.raises(MissingValuesError, match="no NEW data"):
        splice(spec, {"OLD": _daily("2019-09-02", "2019-09-30", 1.0)}, date(2019, 10, 31))


def test_daily_fill_limit_boundary():
    # A gap of DAILY_FILL_DAYS missing days is carried; one more day is not.
    spec = _spec(RateSegment("X", "daily"))
    first = pd.Timestamp("2024-01-10")

    def two_obs(missing_days: int) -> pd.Series:
        nxt = first + pd.Timedelta(days=missing_days + 1)
        return pd.Series(1.0, index=pd.DatetimeIndex([first, nxt]))

    ok = two_obs(DAILY_FILL_DAYS)
    assert not splice(spec, {"X": ok}, ok.index[-1].date()).isna().any()
    bad = two_obs(DAILY_FILL_DAYS + 1)
    with pytest.raises(MissingValuesError):
        splice(spec, {"X": bad}, bad.index[-1].date())


def test_config_validation():
    fred = {"A": {"frequency": "daily"}, "B": {"frequency": "daily"}, "W": {"frequency": "weekly"}}

    def specs(ccy_cfg):
        return parse_specs({"fred": fred, "short_rates": {"EUR": ccy_cfg}})

    ok = specs(
        {
            "day_count": 360,
            "segments": [{"series": "A", "until": date(2020, 1, 1)}, {"series": "B"}],
        }
    )
    assert ok["EUR"].segments[0].until == date(2020, 1, 1)
    bad = [
        {"day_count": 252, "segments": [{"series": "A"}]},
        {"day_count": 360, "segments": []},
        {"day_count": 360, "segments": [{"series": "NOPE"}]},
        {"day_count": 360, "segments": [{"series": "W"}]},
        {"day_count": 360, "segments": [{"series": "A"}, {"series": "B"}]},
        {"day_count": 360, "segments": [{"series": "A", "until": date(2020, 1, 1)}]},
        {"day_count": 360, "segments": [{"series": "A", "until": "2020-01-01"}, {"series": "B"}]},
        {
            "day_count": 360,
            "segments": [
                {"series": "A", "until": date(2021, 1, 1)},
                {"series": "B", "until": date(2020, 1, 1)},
                {"series": "A"},
            ],
        },
    ]
    for cfg in bad:
        with pytest.raises(ConfigError):
            specs(cfg)


def test_repo_config_is_valid_and_covers_the_fx_currencies(settings):
    specs = parse_specs(settings.instruments())
    assert set(specs) == {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NOK", "NZD", "SEK"}
    assert specs["GBP"].day_count == 365 and specs["USD"].day_count == 360


def test_health_check_reports_a_gap(settings):
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    keep = days[(days < "2024-02-01") | (days > "2024-02-20")]
    obs = pd.DataFrame({"series_id": "DFF", "date": keep, "value": 5.0})
    rows = pd.DataFrame(check_short_rates(obs, settings))
    usd = rows[rows["subject"] == "USD"].iloc[0]
    assert usd["status"] == "FAIL" and "USD" in usd["detail"]
    others = rows[rows["subject"] != "USD"]
    assert (others["status"] == "FAIL").all()  # no data at all for the other currencies
