"""Pillar 4 study: popular trading rules after overfitting correction.

``run_development`` runs every pre-registered trial on data up to DEV_END (the holdout is never
loaded), runs the six tests, re-runs a 5% sample on the loop engine, and writes
``reports/pillar4/development/``: the trial table, per-asset results, a JSON summary and the run
manifest. The README section and figures are made by ``pillar4_report``.

The holdout run is separate and deliberately not part of this module's default path.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_research.backtest import loop_engine, weights_engine
from quant_research.backtest.config import load_config
from quant_research.backtest.strategy import FixedWeights
from quant_research.backtest.trial_log import run_manifest
from quant_research.data.settings import Settings, load_settings
from quant_research.studies import pillar4_tests as T
from quant_research.studies.pillar4_grid import CURRENCIES, DEV_END, ETFS, SEED, rules_for, trial_id
from quant_research.studies.pillar4_trials import asset_market, load, run_all, save

log = logging.getLogger(__name__)
ASSETS = ETFS + CURRENCIES
ENGINE_SAMPLE = 0.05


def report_dir(settings: Settings, stage: str = "development") -> Path:
    return settings.root / "reports" / "pillar4" / stage


def cache_dir(settings: Settings, stage: str = "development") -> Path:
    return settings.data_dir / "studies" / "pillar4" / stage


def _engine_gap(tid: str, end: str) -> tuple[str, float]:
    settings = load_settings()
    cfg = replace(load_config(settings), lookahead_checks=0)
    asset = tid.split("|")[0]
    rule = next(r for r in rules_for(asset) if trial_id(r) == tid)
    market = asset_market(settings, asset, end)
    a = weights_engine.run(rule, market, cfg).nav.to_numpy()
    b = loop_engine.run(rule, market, cfg).nav.to_numpy()
    return tid, float(np.max(np.abs(a / b - 1.0)))


def engine_check(trial_ids: list[str], end: str) -> pd.Series:
    """Largest relative NAV gap between the two engines for a seeded 5% sample of trials."""
    rng = np.random.default_rng(SEED)
    n = max(1, round(ENGINE_SAMPLE * len(trial_ids)))
    sample = sorted(rng.choice(trial_ids, size=n, replace=False).tolist())
    with ProcessPoolExecutor() as pool:
        gaps = dict(pool.map(_engine_gap, sample, [end] * n))
    return pd.Series(gaps, name="max relative NAV gap")


def benchmark_stats(settings: Settings, table: pd.DataFrame) -> pd.DataFrame:
    """Per asset: buy-and-hold over the evaluation period, and the share of trials beating it.

    Added after seeing results (Deviations, 2026-11-26) to show where positive active returns
    come from. For currencies "buy-and-hold" is holding the currency (spot plus carry).
    """
    cfg = replace(load_config(settings), lookahead_checks=0)
    rows = {}
    for asset, grp in table.groupby("asset"):
        market = asset_market(settings, str(asset), DEV_END)
        hold = weights_engine.run(FixedWeights({str(asset): 1.0}), market, cfg)
        nav = hold.nav.loc[pd.Timestamp(str(grp["start"].iloc[0])) :]
        years = (nav.index[-1] - nav.index[0]).days / 365.25
        rows[str(asset)] = {
            "buy-and-hold CAGR": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
            "share of trials beating benchmark": float((grp["next_open x1"] > 0).mean()),
        }
    return pd.DataFrame(rows).T


def run_development(settings: Settings, recompute: bool = True) -> dict[str, Any]:
    cache = cache_dir(settings)
    if recompute:
        log.info("running %d assets", len(ASSETS))
        save(run_all(ASSETS, DEV_END), cache)
    results = load(ASSETS, cache)
    table = T.trial_table(results)
    summary: dict[str, Any] = {"period_end": DEV_END, "trials": len(table)}
    summary["h1_h2"] = T.h1_h2(results, table)
    summary["h3"] = T.h3(results, table)
    per_asset = T.h4_h5(results).join(benchmark_stats(settings, table))
    summary["h6"] = T.h6(table).to_dict()
    gaps = engine_check(table.index.tolist(), DEV_END)
    summary["engine_check"] = {"trials": len(gaps), "max_gap": float(gaps.max())}

    out = report_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "trials.csv")
    per_asset.to_csv(out / "per_asset.csv")
    gaps.to_csv(out / "engine_check.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    manifest = run_manifest(settings, {"stage": "development", "period_end": DEV_END})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary
