"""The pre-registered rule families on hand-built prices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bt_helpers import dates, make_market, random_market

from quant_research.backtest import rules as R
from quant_research.backtest.lookahead import check_no_lookahead
from quant_research.backtest.market import Market
from quant_research.errors import InvalidParameterError


def bars(close, opens=None, high=None, low=None) -> Market:
    idx = dates(len(close))
    c = pd.DataFrame({"A": close}, index=idx, dtype=float)
    o = pd.DataFrame({"A": opens if opens is not None else close}, index=idx, dtype=float)
    h = pd.DataFrame({"A": high if high is not None else np.maximum(c["A"], o["A"])}, index=idx)
    lo = pd.DataFrame({"A": low if low is not None else np.minimum(c["A"], o["A"])}, index=idx)
    m = make_market(c, o)
    return Market(
        open=m.open,
        close=m.close,
        asset_class=m.asset_class,
        asset_rate=m.asset_rate,
        asset_day_count=m.asset_day_count,
        cash_rate=m.cash_rate,
        high_prices=h.astype(float),
        low_prices=lo.astype(float),
    )


def pos(rule: R.Rule, market: Market) -> list[float]:
    return rule.target_weights(market)[rule.asset].tolist()


def test_price_vs_ma_and_momentum():
    m = bars([1, 2, 3, 2, 1, 2, 3])
    assert pos(R.PriceVsMA("A", n=3), m) == [0, 0, 1, 0, 0, 1, 1]
    assert pos(R.PriceVsMA("A", allow_short=True, n=3), m) == [0, 0, 1, -1, -1, 1, 1]
    assert pos(R.TimeSeriesMomentum("A", lookback=2), m) == [0, 0, 1, 0, 0, 0, 1]


def test_channel_breakout_keeps_its_position_inside_the_channel():
    m = bars([5, 5, 6, 5.5, 5.8, 4, 4.5, 7])
    # n = 2: long on 6 (> max(5, 5)); hold; short/flat on 4 (< min(5.5, 5.8)); long on 7
    assert pos(R.ChannelBreakout("A", allow_short=True, n=2), m) == [0, 0, 1, 1, 1, -1, -1, 1]


def test_filter_rule():
    m = bars([100, 104, 106, 103, 100.6, 101, 106])
    # 5%: long at 106 (>= 100 x 1.05); out at 100.6 (<= 106 x 0.95);
    # back in at 106 (>= 100.6 x 1.05)
    assert pos(R.FilterRule("A", allow_short=True, x=0.05), m) == [0, 0, 1, 1, -1, -1, 1]
    assert pos(R.FilterRule("A", x=0.05), m) == [0, 0, 1, 1, 0, 0, 1]


def test_rsi_values_and_hysteresis():
    up = pd.Series(np.arange(1.0, 30.0))
    assert R.wilder_rsi(up, 14).iloc[-1] == 100.0
    zigzag = pd.Series(100 + np.tile([1.0, -1.0], 40).cumsum())
    assert R.wilder_rsi(zigzag, 14).iloc[-1] == pytest.approx(50, abs=3)
    m = bars(list(np.r_[np.linspace(100, 80, 20), np.linspace(80, 100, 20)]))
    p = pos(R.RSIRule("A", period=7, lower=30, upper=70), m)
    first = p.index(1.0)
    assert first < 20 and p[-1] == 0.0  # bought into the fall, sold once RSI passed 50


def test_bollinger_enters_below_the_band_and_exits_at_the_mean():
    m = bars([10.0] * 20 + [8.0, 9.0, 10.5, 10.5])
    p = pos(R.Bollinger("A", n=20, k=2.0), m)
    assert p[19] == 0 and p[20] == 1 and p[22] == 0  # entry on the drop, exit above the mean


def test_parabolic_sar_flips_on_a_break():
    rise = np.linspace(100, 120, 30)
    close = np.r_[rise, 110.0, 105.0]
    m = bars(close, high=close + 0.5, low=close - 0.5)
    p = pos(R.ParabolicSAR("A"), m)
    assert all(x == 1 for x in p[:30]) and p[-1] == 0


def test_heikin_ashi_needs_k_rising_candles():
    close = [10, 11, 12, 13, 12, 11, 10]
    m = bars(close, opens=[10, 10, 11, 12, 13, 12, 11])
    p = pos(R.HeikinAshi("A", k=2), m)
    assert p[0] == 0 and p[2] == 1 and p[-1] == 0


def test_shooting_star_steps_aside_for_hold_days():
    n = 30
    close = np.linspace(100, 110, n)
    # ordinary candles: body 0.1, upper shadow 0.05, lower shadow 0.05 (not stars)
    opens, high, low = close - 0.1, close + 0.05, close - 0.15
    high[25] = close[25] + 3.0  # long upper shadow after a rise: a shooting star
    m = bars(close, opens=opens, high=high, low=low)
    p = pos(R.ShootingStar("A", ratio=2.0, hold_days=3), m)
    assert p[24] == 1 and p[25:28] == [0, 0, 0] and p[28] == 1


def test_dual_thrust_and_stochastic():
    flat = [100.0] * 6
    m = bars(flat + [110.0], opens=flat + [100.0], high=[101.0] * 6 + [110.0], low=[99.0] * 7)
    assert pos(R.DualThrust("A", n=5, k=0.5), m)[-1] == 1  # close 110 > open 100 + 0.5 x 2
    close = list(np.r_[np.linspace(100, 90, 20), np.linspace(90, 95, 5)])
    m = bars(close, high=np.array(close) + 1, low=np.array(close) - 1)
    assert pos(R.Stochastic("A", n=14), m)[-1] == 1  # %K rising above %D after a fall


def test_every_rule_is_free_of_look_ahead_and_respects_direction():
    m = random_market(21, n=320, n_etf=1, n_fx=0)
    c = m.close.ffill().bfill()
    m = bars(c["E0"].to_numpy(), opens=m.open.ffill().bfill()["E0"].to_numpy())
    examples = [
        R.MACrossover("A", fast=5, slow=20),
        R.PriceVsMA("A", n=50),
        R.ChannelBreakout("A", n=20),
        R.FilterRule("A", x=0.02),
        R.TimeSeriesMomentum("A", lookback=21),
        R.MACD("A"),
        R.RSIRule("A", period=7),
        R.Bollinger("A", n=20, k=1.5),
        R.ParabolicSAR("A"),
        R.HeikinAshi("A", k=1),
        R.AwesomeOscillator("A"),
        R.ShootingStar("A"),
        R.DualThrust("A", n=5, k=0.3),
        R.Stochastic("A"),
    ]
    assert len({type(r).family for r in examples}) == 14
    for rule in examples:
        check_no_lookahead(rule, m, 40)
        assert min(pos(rule, m)) >= 0  # long/flat by default


def test_ohlc_rules_refuse_currencies():
    m = make_market(pd.DataFrame({"EUR": 1.1}, index=dates(30)), fx=["EUR"])
    with pytest.raises(InvalidParameterError):
        R.ParabolicSAR("EUR", allow_short=True).target_weights(m)
