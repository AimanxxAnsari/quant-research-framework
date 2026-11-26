"""Run a strategy on either engine, or on both engines and both fill rules side by side."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from quant_research.backtest import loop_engine, weights_engine
from quant_research.backtest.config import FILL_RULES, BacktestConfig, FillRule
from quant_research.backtest.market import Market
from quant_research.backtest.result import BacktestResult
from quant_research.backtest.strategy import Strategy
from quant_research.errors import ConfigError

Engine = Literal["weights", "loop"]
ENGINES: dict[str, Callable[[Strategy, Market, BacktestConfig], BacktestResult]] = {
    "weights": weights_engine.run,
    "loop": loop_engine.run,
}


def run(
    strategy: Strategy, market: Market, config: BacktestConfig, engine: Engine = "weights"
) -> BacktestResult:
    if engine not in ENGINES:
        raise ConfigError(f"engine must be one of {sorted(ENGINES)}, got {engine!r}")
    return ENGINES[engine](strategy, market, config)


def label(result: BacktestResult) -> str:
    return f"{result.engine} / {result.fill}"


def run_matrix(
    strategy: Strategy,
    market: Market,
    config: BacktestConfig,
    engines: Sequence[Engine] = ("weights", "loop"),
    fills: Sequence[FillRule] = FILL_RULES,
) -> dict[str, BacktestResult]:
    """Every engine x fill rule combination, keyed like ``"loop / next_open"``."""
    out = {}
    for fill in fills:
        for engine in engines:
            res = run(strategy, market, config.with_fill(fill), engine)
            out[label(res)] = res
    return out


def engine_gap(a: BacktestResult, b: BacktestResult) -> float:
    """Largest relative difference in NAV between two runs on the same dates."""
    if not a.nav.index.equals(b.nav.index):
        raise ConfigError("runs cover different dates")
    return float(np.max(np.abs(a.nav.to_numpy() / b.nav.to_numpy() - 1.0)))


def engine_agreement(results: Mapping[str, BacktestResult]) -> pd.DataFrame:
    """For each fill rule run on both engines: the largest relative NAV gap between them."""
    rows = []
    by_fill: dict[str, dict[str, BacktestResult]] = {}
    for res in results.values():
        by_fill.setdefault(res.fill, {})[res.engine] = res
    for fill, runs in by_fill.items():
        if {"weights", "loop"} <= set(runs):
            gap = engine_gap(runs["weights"], runs["loop"])
            rows.append({"fill": fill, "max relative NAV gap": gap})
    return pd.DataFrame(rows).set_index("fill") if rows else pd.DataFrame()
