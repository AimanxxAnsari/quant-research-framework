from datetime import date

import pandas as pd
from conftest import fred_payload, french_payload, tiingo_payload, treasury_payload

from quant_research.data.build import build
from quant_research.data.quality import check_observations, check_prices
from quant_research.data.raw_store import RawStore


def _populate(settings, ten_year="4.10", closes=(100.0, 101.0)):
    store = RawStore(settings.raw_dir)
    store.save(
        "fred",
        "DGS10",
        fred_payload({"2024-01-02": "3.95", "2024-01-03": "4.00"}),
        url="u",
        ext="json",
    )
    store.save("treasury", "nominal_2024", treasury_payload(ten_year), url="u", ext="csv")
    store.save("french", "F-F_Research_Data_Factors_daily", french_payload(), url="u", ext="zip")
    store.save("tiingo", "SPY", tiingo_payload(closes), url="u", ext="json")
    return store


def _tables(settings):
    c = settings.clean_dir
    return [pd.read_parquet(c / f"{n}.parquet") for n in ("observations", "prices", "series")]


def test_build_is_idempotent(settings):
    _populate(settings)
    build(settings)
    first = _tables(settings)
    build(settings)
    second = _tables(settings)
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)
    assert settings.warehouse.exists()


def test_lineage_columns_present(settings):
    _populate(settings)
    build(settings)
    obs, prices, series = _tables(settings)
    assert {"source_id", "retrieved_at", "raw_sha256"} <= set(obs.columns)
    assert set(series["source_id"]) == {"fred", "treasury", "french"}


def test_planted_bad_value_fails_range_check(settings):
    _populate(settings, ten_year="99.0")  # a 99% 10Y yield is a data error
    build(settings)
    obs, prices, series = _tables(settings)
    res = pd.DataFrame(check_observations(obs, series, settings, as_of=date(2024, 1, 5)))
    bad = res[(res["check"] == "range") & (res["status"] == "FAIL")]
    assert bad["subject"].tolist() == ["UST_PAR_10Y"]


def test_coverage_fails_for_missing_configured_series(settings):
    _populate(settings)
    build(settings)
    obs, prices, series = _tables(settings)
    res = pd.DataFrame(check_observations(obs, series, settings, as_of=date(2024, 1, 5)))
    cov = res[(res["check"] == "coverage") & (res["subject"] == "fred series")]
    assert cov["status"].item() == "FAIL"  # only DGS10 present in the fixture


def test_price_checks_catch_inconsistent_bar_and_jump_and_staleness(settings):
    _populate(settings, closes=(100.0, 250.0))  # +150% in a day
    build(settings)
    _, prices, _ = _tables(settings)
    prices.loc[0, "low"] = 200.0  # low above close
    res = pd.DataFrame(check_prices(prices, ["SPY", "QQQ"], as_of=date(2024, 3, 1)))
    status = {(r.check, r.subject): r.status for r in res.itertuples()}
    assert status[("coverage", "tickers")] == "FAIL"  # QQQ missing
    assert status[("ohlc", "all tickers")] == "FAIL"
    assert status[("jumps", "SPY")] == "WARN"
    assert status[("freshness", "SPY")] == "WARN"
