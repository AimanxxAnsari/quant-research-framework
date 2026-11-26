"""Market validation, configuration, the moving-average rule and loading from the warehouse."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bt_helpers import dates, make_market
from conftest import fred_payload, tiingo_payload

from quant_research.backtest.config import (
    BacktestConfig,
    CostModel,
    config_from_mapping,
    load_config,
)
from quant_research.backtest.data import load_market
from quant_research.backtest.strategy import FixedWeights, MovingAverageCrossover
from quant_research.data.build import build
from quant_research.data.raw_store import RawStore
from quant_research.errors import (
    ConfigError,
    EmptySeriesError,
    InvalidParameterError,
    MisalignedDataError,
    MissingValuesError,
)


def _close(values) -> pd.DataFrame:
    return pd.DataFrame({"A": values}, index=dates(len(values)))


def test_market_rejects_bad_data():
    with pytest.raises(MissingValuesError, match="after it started"):
        make_market(_close([np.nan, 100.0, np.nan, 101.0]))  # a gap after listing
    with pytest.raises(InvalidParameterError):
        make_market(_close([100.0, 0.0, 101.0]))
    with pytest.raises(EmptySeriesError):
        make_market(_close([100.0]))
    c = _close([100.0, 101.0, 102.0])
    with pytest.raises(MisalignedDataError):
        make_market(c, opens=c.iloc[::-1])
    with pytest.raises(MissingValuesError):
        make_market(c, opens=c.where(c > 100.5))  # open missing where close exists
    with pytest.raises(InvalidParameterError):
        make_market(c, asset_rate={"A": 1.0})  # an ETF does not earn interest
    with pytest.raises(MissingValuesError):
        make_market(c, cash_rate=pd.Series([1.0, np.nan, 1.0], index=c.index))


def test_late_listing_is_allowed_and_up_to_cuts_at_the_close():
    m = make_market(_close([np.nan, np.nan, 100.0, 101.0, 102.0]))
    view = m.up_to(m.dates[2])
    assert list(view.dates) == list(m.dates[:3])
    w = FixedWeights({"A": 1.0}).target_weights(m)
    assert w["A"].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]  # flat until listed


def test_accrual_uses_calendar_days_and_the_previous_rate():
    idx = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])  # Friday, Monday, Tuesday
    c = pd.DataFrame({"EUR": 1.1}, index=idx)
    m = make_market(
        c,
        fx=["EUR"],
        asset_rate={"EUR": pd.Series([3.65, 7.3, 0.0], index=idx)},
        cash_rate=pd.Series([3.6, 0.0, 0.0], index=idx),
        day_counts={"EUR": 365},
    )
    cash, assets = m.accrual_factors()
    assert cash[1] == pytest.approx(1 + 0.036 * 3 / 360)  # Friday's rate over three days
    assert assets[1, 0] == pytest.approx(1 + 0.0365 * 3 / 365)
    assert assets[2, 0] == pytest.approx(1 + 0.073 * 1 / 365)  # Monday's rate, one day


def test_moving_average_signal_by_hand():
    m = make_market(_close([1.0, 2, 3, 4, 5, 4, 3, 2, 1]))
    long_flat = MovingAverageCrossover("A", 2, 3).target_weights(m)["A"].tolist()
    long_short = MovingAverageCrossover("A", 2, 3, allow_short=True).target_weights(m)["A"].tolist()
    # fast(2) vs slow(3): rising -> above; on the way down the fast average falls below first
    assert long_flat == [0.0, 0.0, 1, 1, 1, 1, 0, 0, 0]
    assert long_short == [0.0, 0.0, 1, 1, 1, 1, -1, -1, -1]
    with pytest.raises(InvalidParameterError):
        MovingAverageCrossover("A", 3, 3)


def test_config_file_loads_and_bad_settings_fail():
    from quant_research.data.settings import load_settings

    cfg = load_config(load_settings())
    assert cfg.fill in ("next_open", "next_close") and cfg.costs.bps["etf"] > 0
    with pytest.raises(ConfigError):
        config_from_mapping({"fill": "same_close"})
    with pytest.raises(ConfigError):
        config_from_mapping({"cost_bps": {"etf": 1}})  # misspelt key
    with pytest.raises(InvalidParameterError):
        CostModel(bps={"etf": -1.0})
    with pytest.raises(ConfigError):
        CostModel(bps={"bond": 1.0})
    with pytest.raises(InvalidParameterError):
        BacktestConfig(initial_nav=0)
    assert CostModel(bps={"etf": 2.0}).scaled(3).fractions(["etf"])[0] == pytest.approx(6e-4)


def test_load_market_from_warehouse(settings):
    store = RawStore(settings.raw_dir)
    days = {"2024-01-02": "5.33", "2024-01-03": "5.33", "2024-01-04": "5.31"}

    def fred(sid, values):
        store.save("fred", sid, fred_payload(values), url="u", ext="json")

    fred("DFF", days)
    fred("EONIARATE", {"2019-09-30": "-0.45"})
    fred("ECBESTRVOLWGTTRMDMNRT", {k: "3.90" for k in days})
    fred("DEXUSEU", {"2024-01-02": "1.10", "2024-01-04": "1.12"})  # no fixing on 3 Jan
    store.save("tiingo", "SPY", tiingo_payload((100.0, 101.0, 102.0)), url="u", ext="json")
    build(settings)

    m = load_market(settings, etfs=["SPY"], fx=["EUR"])
    assert m.assets == ["SPY", "EUR"]
    assert m.asset_class == {"SPY": "etf", "EUR": "fx"}
    assert m.close["EUR"].tolist() == [1.10, 1.10, 1.12]  # missing fixing carried one day
    assert m.asset_rate["EUR"].tolist() == [3.90] * 3 and m.asset_rate["SPY"].eq(0).all()
    assert m.cash_rate.tolist() == [5.33, 5.33, 5.31]
    assert m.asset_day_count["EUR"] == 360
    with pytest.raises(ConfigError):
        load_market(settings, fx=["XYZ"])
