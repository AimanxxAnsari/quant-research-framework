"""Backtest metrics on hand-checkable paths, and the cost sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bt_helpers import RandomWeights, dates, make_market, random_market

from quant_research.backtest.config import BacktestConfig, CostModel
from quant_research.backtest.runner import run
from quant_research.backtest.strategy import FixedWeights
from quant_research.backtest.summary import cost_sweep, summary

NO_COSTS = CostModel(bps={"etf": 0.0, "fx": 0.0})


def test_metrics_on_a_known_path():
    # Bought at close 1 (120): NAV ratios 1, 1, 0.75, 130/120. Rates are zero.
    m = make_market(pd.DataFrame({"A": [100.0, 120, 90, 130]}, index=dates(4)))
    cfg = BacktestConfig(fill="next_close", costs=NO_COSTS)
    s = summary(run(FixedWeights({"A": 1.0}), m, cfg), m)
    assert s["total return"] == pytest.approx(130 / 120 - 1)
    assert s["max drawdown"] == pytest.approx(-0.25)
    assert s["exposure"] == pytest.approx(2 / 3)  # held into days 2 and 3, not into day 1
    assert s["hit rate"] == pytest.approx(0.5)  # one loss, one gain
    r = np.array([0.0, -0.25, 130 / 90 - 1])
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert s["Sortino"] == pytest.approx(r.mean() / downside * np.sqrt(252))
    assert s["Calmar"] == pytest.approx(s["CAGR"] / 0.25)


def test_all_cash_scores_zero_on_risk_adjusted_metrics():
    m = make_market(pd.DataFrame({"A": 100.0}, index=dates(50)), cash_rate=4.0)
    s = summary(run(FixedWeights(), m, BacktestConfig()), m)
    assert s["total return"] > 0  # it earns the cash rate...
    assert s["Sharpe (excess of cash)"] == 0 and s["exposure"] == 0  # ...but no excess
    assert s["average gross leverage"] == 0 and s["turnover per year"] == 0


def test_cost_sweep_scales_costs_and_lowers_sharpe():
    m = random_market(8)
    table = cost_sweep(RandomWeights(8), m, BacktestConfig(costs=CostModel({"etf": 5, "fx": 2})))
    assert list(table.columns) == ["costs x0", "costs x1", "costs x2", "costs x3"]
    drag = table.loc["cost drag per year"]
    assert drag["costs x0"] == 0
    assert drag["costs x2"] == pytest.approx(2 * drag["costs x1"], rel=0.02)
    assert drag["costs x3"] == pytest.approx(3 * drag["costs x1"], rel=0.02)
    sharpe = table.loc["Sharpe (excess of cash)"].to_numpy()
    assert (np.diff(sharpe) < 0).all()
