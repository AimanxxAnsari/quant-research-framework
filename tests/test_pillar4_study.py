"""The Pillar 4 study set-up matches its pre-registration, and its tests run end to end."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from quant_research.backtest.trial_log import run_manifest
from quant_research.studies import pillar4_tests as T
from quant_research.studies.pillar4_grid import (
    CURRENCIES,
    DEV_END,
    ETFS,
    HOLDOUT_START,
    rules_for,
    trial_id,
)
from quant_research.studies.pillar4_trials import VARIANTS, AssetTrials

PREREGISTERED_PER_ASSET = {  # PREREGISTRATION.md, section 3
    "MA crossover": 11,
    "price vs MA": 3,
    "channel breakout": 4,
    "filter rule": 4,
    "time-series momentum": 4,
    "MACD": 3,
    "RSI": 4,
    "Bollinger bands": 6,
    "Parabolic SAR": 3,
    "Heikin-Ashi": 3,
    "Awesome Oscillator": 2,
    "shooting star": 6,
    "Dual Thrust (daily)": 6,
    "stochastic oscillator": 2,
}


def test_grid_matches_the_preregistration():
    assert len(ETFS) == 33 and len(CURRENCIES) == 9
    etf = Counter(r.family for r in rules_for("SPY"))
    fx = Counter(r.family for r in rules_for("EUR"))
    assert etf == PREREGISTERED_PER_ASSET
    close_based = list(PREREGISTERED_PER_ASSET)[:8]  # families 1-8 use closes only
    assert fx == {k: PREREGISTERED_PER_ASSET[k] for k in close_based}
    assert sum(fx.values()) == 39 and sum(etf.values()) == 61
    all_ids = [trial_id(r) for a in ETFS + CURRENCIES for r in rules_for(a)]
    assert len(all_ids) == len(set(all_ids)) == 2364
    assert all(not r.allow_short for r in rules_for("SPY"))
    assert all(r.allow_short for r in rules_for("JPY"))
    assert DEV_END == "2021-11-30" and HOLDOUT_START == "2021-12-01"


def _fake(asset: str, cls: str, skill: int | None, seed: int) -> AssetTrials:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=1600)
    ids = [f"{asset}|rule|p={i}" for i in range(6)]
    data = rng.normal(0, 0.005, (len(idx), 6))
    if skill is not None:
        data[:, skill] += 0.002
    sharpe = pd.DataFrame(
        {v: data.mean(axis=0) / data.std(axis=0, ddof=1) * np.sqrt(252) for v in VARIANTS},
        index=ids,
    )
    return AssetTrials(asset, cls, pd.DataFrame(data, index=idx, columns=ids), sharpe)


def test_six_tests_run_and_find_a_planted_skilled_trial():
    results = [_fake("A", "etf", 2, 1), _fake("B", "etf", None, 2), _fake("C", "fx", None, 3)]
    table = T.trial_table(results)
    h12 = T.h1_h2(results, table)
    assert h12["etf"]["spa_p"] < 0.01 and "A|rule|p=2" in h12["etf"]["stepm_survivors"]
    assert "A|rule|p=2" in h12["bhy_survivors"]
    h3 = T.h3(results, table)
    assert h3["etf"]["best"] == "A|rule|p=2" and h3["etf"]["passes"]
    assert not h3["fx"]["passes"]
    per_asset = T.h4_h5(results)
    assert per_asset.loc["A", "PBO"] < 0.1
    assert per_asset.loc["A", "walk-forward OOS Sharpe"] > 2
    assert set(T.h6(table).index) == {"etf", "fx"}


def test_manifest_records_code_and_data(settings):
    settings.clean_dir.mkdir(parents=True)
    pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-05"])}).to_parquet(
        settings.clean_dir / "prices.parquet"
    )
    m = run_manifest(settings, {"stage": "test"})
    assert m["data_through"] == "2026-01-05" and m["stage"] == "test"
    assert len(m["data_sha256"]["prices.parquet"]) == 64
    assert len(m["git_commit"]) == 40
