"""Run every pre-registered trial and keep its daily active returns.

Active return (PREREGISTRATION.md section 5): ETF trial excess return over cash minus the
excess return of buying and holding the same ETF under the same fill rule and costs; for a
currency the benchmark is cash, so active = excess. The first ``WARMUP_DAYS`` trading days of
each asset are not evaluated.

Variants per trial: the headline (next-open fills, 1x costs, look-ahead check on), next-close
fills, and 0x / 2x / 3x costs. Only the headline keeps the full daily series; the others keep
their annualised active Sharpe ratio.

Daily series are derived from vendor prices, so they are cached under ``data/`` (git-ignored)
and never committed.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pandas as pd

from quant_research.backtest import weights_engine
from quant_research.backtest.config import BacktestConfig, FillRule, load_config
from quant_research.backtest.data import load_market
from quant_research.backtest.market import Market
from quant_research.backtest.result import BacktestResult
from quant_research.backtest.strategy import FixedWeights
from quant_research.backtest.summary import excess_returns
from quant_research.data import rates
from quant_research.data.settings import Settings, load_settings
from quant_research.studies.pillar4_grid import CURRENCIES, WARMUP_DAYS, rules_for, trial_id

HEADLINE = "next_open x1"
VARIANTS: dict[str, tuple[str, float]] = {
    HEADLINE: ("next_open", 1.0),
    "next_close x1": ("next_close", 1.0),
    "next_open x0": ("next_open", 0.0),
    "next_open x2": ("next_open", 2.0),
    "next_open x3": ("next_open", 3.0),
}
TRADING_DAYS = 252


@dataclass(frozen=True)
class AssetTrials:
    asset: str
    asset_class: str
    active: pd.DataFrame  # evaluated dates x trial ids, headline variant
    sharpe: pd.DataFrame  # trial ids x variants, annualised active Sharpe


def asset_market(settings: Settings, asset: str, end: str | None) -> Market:
    """One-asset market from the first date with both prices and short rates."""
    if asset in CURRENCIES:
        first_rate = rates.short_rates(settings, ["USD", asset]).apply(
            lambda c: c.first_valid_index()
        )
        start = f"{max(first_rate):%Y-%m-%d}"
        return load_market(settings, fx=[asset], start=start, end=end)
    return load_market(settings, etfs=[asset], end=end)


def _variant_config(base: BacktestConfig, variant: str) -> BacktestConfig:
    fill, mult = VARIANTS[variant]
    checks = base.lookahead_checks if variant == HEADLINE else 0
    return replace(
        base.with_fill(cast(FillRule, fill)), costs=base.costs.scaled(mult), lookahead_checks=checks
    )


def _active(res: BacktestResult, bench: np.ndarray, market: Market) -> pd.Series:
    ex = excess_returns(res, market) - bench
    return pd.Series(ex, index=market.dates[1:])


def annual_sharpe(r: np.ndarray) -> float:
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def run_asset(asset: str, end: str | None, start_eval: str | None = None) -> AssetTrials:
    """All trials on one asset. Evaluated from ``start_eval`` if given, else after warm-up."""
    settings = load_settings()
    base = load_config(settings)
    market = asset_market(settings, asset, end)
    is_fx = asset in CURRENCIES
    rules = rules_for(asset)
    ids = [trial_id(r) for r in rules]
    keep = market.dates[1:][WARMUP_DAYS:] if start_eval is None else market.dates[1:]
    if start_eval is not None:
        keep = keep[keep >= pd.Timestamp(start_eval)]
    active: dict[str, pd.Series] = {}
    sharpe = pd.DataFrame(index=ids, columns=list(VARIANTS), dtype=float)
    for variant in VARIANTS:
        cfg = _variant_config(base, variant)
        bench = np.zeros(len(market.dates) - 1)
        if not is_fx:
            hold = weights_engine.run(FixedWeights({asset: 1.0}), market, cfg)
            bench = excess_returns(hold, market)
        for rule, tid in zip(rules, ids, strict=True):
            series = _active(weights_engine.run(rule, market, cfg), bench, market).loc[keep]
            sharpe.loc[tid, variant] = annual_sharpe(series.to_numpy())
            if variant == HEADLINE:
                active[tid] = series
    return AssetTrials(asset, "fx" if is_fx else "etf", pd.DataFrame(active), sharpe)


def run_all(
    assets: tuple[str, ...], end: str | None, start_eval: str | None = None, workers: int = 0
) -> list[AssetTrials]:
    n = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(run_asset, a, end, start_eval) for a in assets]
        return [f.result() for f in futures]


def save(results: list[AssetTrials], folder: os.PathLike[str] | str) -> None:
    path = os.fspath(folder)
    os.makedirs(path, exist_ok=True)
    for r in results:
        r.active.to_parquet(os.path.join(path, f"{r.asset}_active.parquet"))
        r.sharpe.to_parquet(os.path.join(path, f"{r.asset}_sharpe.parquet"))


def load(assets: tuple[str, ...], folder: os.PathLike[str] | str) -> list[AssetTrials]:
    path = os.fspath(folder)
    out = []
    for a in assets:
        active = pd.read_parquet(os.path.join(path, f"{a}_active.parquet"))
        sharpe = pd.read_parquet(os.path.join(path, f"{a}_sharpe.parquet"))
        out.append(AssetTrials(a, "fx" if a in CURRENCIES else "etf", active, sharpe))
    return out
