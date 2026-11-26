"""The Streamlit app: its logic on the synthetic market, and the pages run headless (AppTest).

Everything uses the demo market or the committed study reports, so no data or keys are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from quant_research.app import charts, lab  # noqa: E402
from quant_research.app.study import filter_trials, load_study  # noqa: E402
from quant_research.backtest.config import BacktestConfig  # noqa: E402
from quant_research.errors import ConfigError, InvalidParameterError  # noqa: E402
from quant_research.studies.pillar4_grid import CURRENCIES, ETFS, rules_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "src" / "quant_research" / "app" / "main.py")


def test_every_study_rule_can_be_rebuilt_from_its_family_and_parameters() -> None:
    for asset in (ETFS[0], CURRENCIES[0]):
        for rule in rules_for(asset):
            rebuilt = lab.build_rule(rule.family, rule.asset, rule.allow_short, rule.params)
            assert rebuilt == rule


def test_families_for_fx_leave_out_the_high_low_rules() -> None:
    fx = lab.families_for("fx")
    assert len(lab.families_for("etf")) == 14 and len(fx) == 8
    assert all(not lab.FAMILIES[f].needs_ohlc for f in fx)


def test_build_rule_rejects_bad_input() -> None:
    with pytest.raises(InvalidParameterError):
        lab.build_rule("MA crossover", "X", False, {"fast": 2.5})
    with pytest.raises(ConfigError):
        lab.build_rule("MA crossover", "X", False, {"window": 3})
    with pytest.raises(ConfigError):
        lab.build_rule("astrology", "X", False, {})


def test_demo_market_is_valid_and_seeded() -> None:
    a, b = lab.demo_market(), lab.demo_market()
    assert a.close.equals(b.close)
    assert a.asset_class == {"TREND": "etf", "DECAY": "etf", "FXA": "fx", "FXB": "fx"}
    decay = a.close["DECAY"]
    assert decay.iloc[-1] < 0.5 * decay.iloc[0]  # the decaying ETF does lose money


@pytest.mark.parametrize(("asset", "family"), [("TREND", "Parabolic SAR"), ("FXA", "MACD")])
def test_both_engines_agree_and_the_comparison_lines_up(asset: str, family: str) -> None:
    market = lab.single_asset(lab.demo_market(n=400), asset)
    rule = lab.build_rule(family, asset, asset == "FXA", {})
    run = lab.run_lab(market, rule, asset, BacktestConfig(), lab.ENGINE_CHOICES["both"])
    assert run.max_gap() < 1e-10
    table = run.comparison()
    assert np.allclose(table["loop minus weights"], 0.0, atol=1e-9)
    assert list(run.metrics_table("loop").columns) == ["strategy", "benchmark"]
    assert run.runs["weights"].benchmark_metrics["active Sharpe"] == 0.0


def test_single_asset_starts_at_the_later_of_listing_and_start() -> None:
    market = lab.demo_market(n=300)
    cut = lab.single_asset(market, "FXA", "2019-06-01")
    assert cut.dates[0] == market.dates[market.dates >= "2019-06-01"][0]
    assert lab.single_asset(market, "FXA", "1990-01-01").dates[0] == market.dates[0]
    with pytest.raises(ConfigError):
        lab.single_asset(market, "FXA", "2030-01-01")


def test_comparison_needs_both_engines() -> None:
    market = lab.single_asset(lab.demo_market(n=300), "TREND")
    rule = lab.build_rule("price vs MA", "TREND", False, {"n": 20})
    run = lab.run_lab(market, rule, "TREND", BacktestConfig(), ("weights",))
    with pytest.raises(ConfigError):
        run.comparison()


def test_charts_build_in_both_themes() -> None:
    market = lab.single_asset(lab.demo_market(n=300), "DECAY")
    rule = lab.build_rule("RSI", "DECAY", False, {})
    run = lab.run_lab(market, rule, "DECAY", BacktestConfig(), ("weights", "loop"))
    for kind in ("light", "dark"):
        t = charts.theme_for(kind)
        for fig in (charts.equity(run, "loop", t), charts.nav_gap(run, t), charts.timings(run, t)):
            assert fig.layout.paper_bgcolor == t.surface


def test_log_ticks_step_one_two_five() -> None:
    assert charts._log_ticks(0.07, 2.3) == [0.1, 0.2, 0.5, 1.0, 2.0]
    assert charts._log_ticks(0.9, 1.1) == [1.0]


def test_study_results_load_and_filter() -> None:
    res = load_study(ROOT)
    assert len(res.trials) == res.summary["trials"]
    ung = filter_trials(res.trials, assets=["UNG"], families=["RSI"])
    assert len(ung) > 0 and set(ung["asset"]) == {"UNG"} and set(ung["family"]) == {"RSI"}
    assert len(filter_trials(res.trials)) == len(res.trials)


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def test_lab_page_runs_on_the_demo_market() -> None:
    at = _app()
    assert not at.exception and not at.error
    assert at.title[0].value == "Engine lab"


def test_engine_toggle_both_shows_the_comparison() -> None:
    at = _app()
    at.radio(key="engine").set_value("both").run()
    assert not at.exception and not at.error
    assert any(h.value == "Engine comparison" for h in at.subheader)
    gap = next(m for m in at.metric if m.label == "Largest NAV gap")
    assert float(gap.value) < 1e-10


def test_currency_rules_offer_shorts_and_close_only_families() -> None:
    at = _app()
    at.selectbox(key="asset").set_value("FXA").run()
    assert not at.exception and not at.error
    assert "Parabolic SAR" not in at.selectbox(key="family").options
    assert at.checkbox(key="short_fx").value


def test_study_page_renders_committed_results() -> None:
    at = _app()
    at.radio(key="page").set_value("Study explorer").run()
    assert not at.exception and not at.error
    assert any(h.value == "Holdout" for h in at.subheader)
