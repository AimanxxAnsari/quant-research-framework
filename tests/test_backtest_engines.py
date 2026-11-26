"""Both backtest engines on known answers, against each other, and against planted bugs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bt_helpers import (
    FromFrame,
    PeekTomorrow,
    RandomWeights,
    dates,
    make_market,
    random_market,
)

from quant_research.backtest.config import BacktestConfig, CostModel
from quant_research.backtest.lookahead import check_no_lookahead
from quant_research.backtest.runner import engine_agreement, run, run_matrix
from quant_research.backtest.strategy import FixedWeights, MovingAverageCrossover
from quant_research.errors import (
    AccountWipedOutError,
    InvalidParameterError,
    LookAheadError,
    MissingValuesError,
)

ENGINES = ["weights", "loop"]
FILLS = ["next_open", "next_close"]
X0 = 1_000_000.0
NO_COSTS = CostModel(bps={"etf": 0.0, "fx": 0.0})


def cfg(fill="next_close", costs=NO_COSTS, **kw) -> BacktestConfig:
    return BacktestConfig(fill=fill, initial_nav=X0, costs=costs, **kw)


def accrual(rate_pct: float, days: np.ndarray, basis: int = 360) -> float:
    return float(np.prod(1 + rate_pct / 100 * days / basis))


# ---------------------------------------------------------------- known answers


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fill", FILLS)
def test_all_cash_earns_the_cash_rate(engine, fill):
    idx = dates(30)
    m = make_market(pd.DataFrame({"A": 100.0}, index=idx), cash_rate=3.65)
    res = run(FixedWeights(), m, cfg(fill), engine)
    days = np.diff(idx.to_numpy()).astype("timedelta64[D]").astype(float)
    assert res.nav.iloc[-1] == pytest.approx(X0 * accrual(3.65, days), rel=1e-13)


@pytest.mark.parametrize("engine", ENGINES)
def test_buy_and_hold_earns_price_change_from_the_fill_price(engine):
    idx = dates(6)
    close = pd.DataFrame({"A": [100.0, 102, 104, 103, 107, 110]}, index=idx)
    opens = pd.DataFrame({"A": [99.0, 101, 103, 105, 106, 108]}, index=idx)
    m = make_market(close, opens)
    s = FixedWeights({"A": 1.0})
    on_close = run(s, m, cfg("next_close"), engine)
    on_open = run(s, m, cfg("next_open"), engine)
    # decided at close 0, so bought at close 1 (102) or open 1 (101); never at close 0 (100)
    assert on_close.nav.iloc[-1] == pytest.approx(X0 * 110 / 102, rel=1e-13)
    assert on_open.nav.iloc[-1] == pytest.approx(X0 * 110 / 101, rel=1e-13)
    assert on_open.nav.iloc[0] == on_close.nav.iloc[0] == X0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fill", FILLS)
def test_round_trip_costs_two_times_the_one_way_cost(engine, fill):
    idx = dates(5)
    m = make_market(pd.DataFrame({"A": 100.0}, index=idx))
    table = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=idx)  # buy, then sell next day
    res = run(FromFrame(table), m, cfg(fill, CostModel(bps={"etf": 10.0, "fx": 0.0})), engine)
    assert res.nav.iloc[-1] == pytest.approx(X0 * (1 - 2 * 0.001), rel=1e-13)
    assert res.costs.sum() == pytest.approx(2 * 0.001 * X0, rel=1e-13)
    assert res.turnover.sum() == pytest.approx(1 + X0 / (X0 * (1 - 0.001)), rel=1e-13)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fill", FILLS)
def test_fx_long_earns_the_foreign_rate_and_short_pays_it(engine, fill):
    idx = dates(40)
    days = np.diff(idx.to_numpy()).astype("timedelta64[D]").astype(float)
    m = make_market(
        pd.DataFrame({"EUR": 1.10}, index=idx), fx=["EUR"], asset_rate={"EUR": 4.0}, cash_rate=1.0
    )
    before_fill = 1 + 0.01 * days[0] / 360  # cash only until the first fill
    long = run(FixedWeights({"EUR": 1.0}), m, cfg(fill), engine)
    assert long.nav.iloc[-1] == pytest.approx(X0 * before_fill * accrual(4.0, days[1:]), rel=1e-12)
    short = run(FixedWeights({"EUR": -1.0}), m, cfg(fill), engine)
    # short 1x EUR: hold 2x NAV in USD cash at 1%, owe 1x NAV in EUR at 4%
    growth = np.prod(2 * (1 + 0.01 * days[1:] / 360) - (1 + 0.04 * days[1:] / 360))
    assert short.nav.iloc[-1] == pytest.approx(X0 * before_fill * growth, rel=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_fx_long_gains_the_spot_move(engine):
    idx = dates(3)
    m = make_market(pd.DataFrame({"EUR": [1.10, 1.10, 1.21]}, index=idx), fx=["EUR"])
    res = run(FixedWeights({"EUR": 1.0}), m, cfg("next_close"), engine)
    assert res.nav.iloc[-1] == pytest.approx(X0 * 1.10, rel=1e-13)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fill", FILLS)
def test_a_decision_never_earns_the_move_that_triggered_it(engine, fill):
    # The price jumps at the close of day 5; a rule that buys on seeing the jump must miss it.
    idx = dates(10)
    close = pd.DataFrame({"A": [100.0] * 5 + [110.0] * 5}, index=idx)
    m = make_market(close)
    table = (close >= 110).astype(float)
    res = run(FromFrame(table, "buy after the jump"), m, cfg(fill), engine)
    assert res.nav.iloc[-1] == pytest.approx(X0, rel=1e-13)


@pytest.mark.parametrize("engine", ENGINES)
def test_leverage_wipe_out_is_an_error_not_a_negative_nav(engine):
    idx = dates(4)
    m = make_market(pd.DataFrame({"A": [100.0, 100, 100, 70]}, index=idx))
    with pytest.raises(AccountWipedOutError):
        run(FixedWeights({"A": 4.0}), m, cfg("next_close"), engine)


# ---------------------------------------------------------------- engines agree


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("fill", FILLS)
def test_engines_agree_exactly_with_costs_carry_and_late_listing(seed, fill):
    m = random_market(seed)
    config = cfg(fill, CostModel(bps={"etf": 5.0, "fx": 2.0}))
    w = run(RandomWeights(seed), m, config, "weights")
    loop = run(RandomWeights(seed), m, config, "loop")
    np.testing.assert_allclose(w.nav, loop.nav, rtol=1e-11)
    np.testing.assert_allclose(w.costs, loop.costs, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(w.turnover, loop.turnover, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(w.weights, loop.weights, rtol=1e-9, atol=1e-12)
    assert w.costs.sum() > 0 and (w.turnover > 0).sum() > 50  # the test really trades


def test_run_matrix_and_agreement_table():
    m = random_market(7)
    results = run_matrix(RandomWeights(7), m, cfg())
    assert set(results) == {f"{e} / {f}" for e in ENGINES for f in FILLS}
    gaps = engine_agreement(results)
    assert (gaps["max relative NAV gap"] < 1e-11).all()


# ---------------------------------------------------------------- look-ahead


def test_planted_look_ahead_is_caught_by_the_weights_engine():
    m = random_market(4)
    with pytest.raises(LookAheadError, match="peek at tomorrow"):
        run(PeekTomorrow("E0"), m, cfg(), "weights")


def test_without_the_check_look_ahead_looks_brilliant_and_the_loop_engine_disagrees():
    m = random_market(4)
    peeking = run(PeekTomorrow("E0"), m, cfg(lookahead_checks=0), "weights")
    honest = run(PeekTomorrow("E0"), m, cfg(), "loop")  # never sees tomorrow, so stays flat
    assert peeking.nav.iloc[-1] / X0 > 3  # an impossible year
    assert honest.weights["E0"].abs().max() == 0


def test_legitimate_strategies_pass_the_look_ahead_check():
    m = random_market(5)
    check_no_lookahead(MovingAverageCrossover("E0", 5, 20, allow_short=True), m, 50)
    check_no_lookahead(RandomWeights(5), m, 50)


# ---------------------------------------------------------------- bad targets


def test_bad_targets_fail_loudly_in_both_engines():
    idx = dates(5)
    close = pd.DataFrame({"A": [np.nan, 100.0, 101, 102, 103], "B": 50.0}, index=idx)
    m = make_market(close)
    cases = [
        (
            pd.DataFrame({"A": 0.0, "B": [0.5, np.nan, 0.5, 0.5, 0.5]}, index=idx),
            MissingValuesError,
        ),
        (pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.5, 0.5], "B": 0.0}, index=idx), MissingValuesError),
        (pd.DataFrame({"A": 0.0, "B": 5.0}, index=idx), InvalidParameterError),
    ]
    for table, err in cases:
        for engine in ENGINES:
            with pytest.raises(err):
                run(FromFrame(table), m, cfg(lookahead_checks=0), engine)
