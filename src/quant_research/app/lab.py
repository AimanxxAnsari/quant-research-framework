"""Logic behind the engine lab page, kept free of Streamlit so it can be tested directly.

The lab runs one rule from the Pillar 4 study on one asset, on the weights engine, the loop
engine, or both, and compares them. Anything chosen here is in-sample exploration: it is not
part of the pre-registered study and says nothing about out-of-sample performance.

Conventions follow ADR-0009 to ADR-0012: decisions at the close, filled at the next open or
close, fixed one-way costs, ETFs benchmarked against buy-and-hold of the same ETF and currencies
against cash. "Active" returns are daily excess returns over cash minus the benchmark's.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import cast

import numpy as np
import pandas as pd

from quant_research.backtest import rules as R
from quant_research.backtest.config import BacktestConfig, FillRule
from quant_research.backtest.market import Market
from quant_research.backtest.result import BacktestResult
from quant_research.backtest.runner import Engine, engine_gap, run
from quant_research.backtest.strategy import FixedWeights, Strategy
from quant_research.backtest.summary import excess_returns, summary
from quant_research.errors import ConfigError, InvalidParameterError
from quant_research.studies.pillar4_trials import annual_sharpe

FAMILIES: dict[str, type[R.Rule]] = {
    cls.family: cls
    for cls in (
        R.MACrossover,
        R.PriceVsMA,
        R.ChannelBreakout,
        R.FilterRule,
        R.TimeSeriesMomentum,
        R.MACD,
        R.RSIRule,
        R.Bollinger,
        R.ParabolicSAR,
        R.HeikinAshi,
        R.AwesomeOscillator,
        R.ShootingStar,
        R.DualThrust,
        R.Stochastic,
    )
}
ENGINE_CHOICES: dict[str, tuple[Engine, ...]] = {
    "weights": ("weights",),
    "loop": ("loop",),
    "both": ("weights", "loop"),
}

# Synthetic market: about six years of business days, so the loop engine stays quick
DEMO_DAYS = 1500
DEMO_SEED = 7


def families_for(asset_class: str) -> list[str]:
    """Rule families that can run on an asset class (high/low rules need ETF bars)."""
    return [f for f, cls in FAMILIES.items() if asset_class == "etf" or not cls.needs_ohlc]


def rule_parameters(family: str) -> dict[str, int | float]:
    """Parameter names and their defaults for one family, in declaration order."""
    cls = _family(family)
    skip = {"asset", "allow_short"}
    return {f.name: cast(int | float, f.default) for f in fields(cls) if f.name not in skip}


def build_rule(family: str, asset: str, allow_short: bool, params: Mapping[str, float]) -> R.Rule:
    """A rule with the given parameters; integer parameters must be whole numbers."""
    defaults = rule_parameters(family)
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        raise ConfigError(f"{family} has no parameters {unknown}")
    typed: dict[str, int | float] = {}
    for name, default in defaults.items():
        value = params.get(name, default)
        if isinstance(default, int):
            if float(value) != int(value) or value <= 0:
                raise InvalidParameterError(f"{name} must be a positive whole number")
            typed[name] = int(value)
        else:
            typed[name] = float(value)
    if _family(family).needs_ohlc and allow_short:
        raise InvalidParameterError(f"{family} runs long/flat on ETFs only")
    return _family(family)(asset, allow_short, **typed)


def _family(family: str) -> type[R.Rule]:
    if family not in FAMILIES:
        raise ConfigError(f"unknown rule family {family!r}")
    return FAMILIES[family]


def demo_market(seed: int = DEMO_SEED, n: int = DEMO_DAYS) -> Market:
    """A seeded synthetic market with OHLC bars and short rates; no vendor data.

    TREND is an ETF that drifts up, DECAY one that loses about 25% a year (like a futures ETF
    with heavy roll costs), and FXA / FXB are currencies with moving short rates. Prices are
    geometric random walks with overnight gaps; high and low extend past open and close.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    etfs = {"TREND": (0.0004, 0.011), "DECAY": (-0.0011, 0.03)}
    cols: dict[str, dict[str, np.ndarray]] = {k: {} for k in ("open", "high", "low", "close")}
    for name, (drift, vol) in etfs.items():
        night = rng.normal(0.0, vol * 0.4, n)
        day = rng.normal(drift, vol, n)
        o = 100.0 * np.exp(np.cumsum(night + np.r_[0.0, day[:-1]]))
        c = o * np.exp(day)
        reach = np.abs(rng.normal(0.0, vol * 0.5, (2, n)))
        cols["open"][name], cols["close"][name] = o, c
        cols["high"][name] = np.maximum(o, c) * np.exp(reach[0])
        cols["low"][name] = np.minimum(o, c) * np.exp(-reach[1])
    fx = ["FXA", "FXB"]
    for name, start in zip(fx, (1.1, 0.7), strict=True):
        spot = start * np.exp(np.cumsum(rng.normal(0.0, 0.006, n)))
        for k in cols:
            cols[k][name] = spot  # one fixing a day, as with the H.10 rates
    frames = {k: pd.DataFrame(v, index=idx) for k, v in cols.items()}
    assets = [*etfs, *fx]
    rates = pd.DataFrame(0.0, index=idx, columns=assets)
    for name, level in zip(fx, (4.0, 0.5), strict=True):
        rates[name] = np.clip(level + np.cumsum(rng.normal(0.0, 0.02, n)), -0.5, None)
    cash = pd.Series(np.clip(2.0 + np.cumsum(rng.normal(0.0, 0.02, n)), 0.0, None), index=idx)
    return Market(
        open=frames["open"][assets],
        close=frames["close"][assets],
        asset_class={a: ("etf" if a in etfs else "fx") for a in assets},
        asset_rate=rates,
        asset_day_count={a: 360 for a in assets},
        cash_rate=cash,
        high_prices=frames["high"][assets].where(frames["close"][assets].notna()),
        low_prices=frames["low"][assets].where(frames["close"][assets].notna()),
    )


def single_asset(market: Market, asset: str, start: str | None = None) -> Market:
    """The market cut down to one asset, from its first price (or ``start``, if later)."""
    if asset not in market.assets:
        raise ConfigError(f"{asset} is not in the market")
    listed = market.close[asset].notna().to_numpy()
    if not listed.any():
        raise ConfigError(f"{asset} has no prices")
    first = market.dates[int(listed.argmax())]
    if start is not None:
        first = max(first, pd.Timestamp(start))
    rows = market.close.index >= first
    if rows.sum() < 2:
        raise ConfigError(f"{asset} has fewer than two dates from {first:%Y-%m-%d}")

    def pick(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        return None if frame is None else frame.loc[rows, [asset]]

    return Market(
        open=market.open.loc[rows, [asset]],
        close=market.close.loc[rows, [asset]],
        asset_class={asset: market.asset_class[asset]},
        asset_rate=market.asset_rate.loc[rows, [asset]],
        asset_day_count={asset: market.asset_day_count[asset]},
        cash_rate=market.cash_rate.loc[rows],
        cash_day_count=market.cash_day_count,
        high_prices=pick(market.high_prices),
        low_prices=pick(market.low_prices),
    )


def benchmark(market: Market, asset: str) -> Strategy:
    """Buy-and-hold for an ETF, cash for a currency (the study's benchmarks)."""
    return FixedWeights({asset: 1.0}) if market.asset_class[asset] == "etf" else FixedWeights({})


@dataclass(frozen=True)
class EngineRun:
    result: BacktestResult
    benchmark: BacktestResult
    seconds: float
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]


@dataclass(frozen=True)
class LabRun:
    asset: str
    asset_class: str
    rule: str
    runs: dict[str, EngineRun]  # engine name -> run
    lookahead_checks: int

    def metrics_table(self, engine: str) -> pd.DataFrame:
        """Strategy and benchmark metrics for one engine, one row per metric."""
        r = self.runs[engine]
        return pd.DataFrame({"strategy": r.metrics, "benchmark": r.benchmark_metrics})

    def comparison(self) -> pd.DataFrame:
        """Weights and loop metrics side by side with their difference (both engines only)."""
        if set(self.runs) != {"weights", "loop"}:
            raise ConfigError("the comparison needs runs on both engines")
        w, lo = self.runs["weights"].metrics, self.runs["loop"].metrics
        table = pd.DataFrame({"weights": w, "loop": lo})
        table["loop minus weights"] = table["loop"] - table["weights"]
        return table

    def nav_gap(self) -> pd.Series:
        """Relative NAV difference loop / weights - 1 on every date (both engines only)."""
        w, lo = self.runs["weights"].result.nav, self.runs["loop"].result.nav
        return lo / w - 1.0

    def max_gap(self) -> float:
        return engine_gap(self.runs["weights"].result, self.runs["loop"].result)


def run_lab(
    market: Market,
    rule: Strategy,
    asset: str,
    config: BacktestConfig,
    engines: Sequence[Engine],
) -> LabRun:
    """Run the rule and its benchmark on each engine, timing the rule's run."""
    bench = benchmark(market, asset)
    runs: dict[str, EngineRun] = {}
    for engine in engines:
        start = time.perf_counter()
        res = run(rule, market, config, engine)
        seconds = time.perf_counter() - start
        bres = run(bench, market, replace(config, lookahead_checks=0), engine)
        metrics = summary(res, market)
        active = excess_returns(res, market) - excess_returns(bres, market)
        metrics["active Sharpe"] = annual_sharpe(active)
        bench_metrics = {**summary(bres, market), "active Sharpe": 0.0}
        runs[engine] = EngineRun(res, bres, seconds, metrics, bench_metrics)
    return LabRun(asset, market.asset_class[asset], rule.name, runs, config.lookahead_checks)


def lab_config(base: BacktestConfig, fill: str, cost_multiplier: float) -> BacktestConfig:
    return replace(base.with_fill(cast(FillRule, fill)), costs=base.costs.scaled(cost_multiplier))
