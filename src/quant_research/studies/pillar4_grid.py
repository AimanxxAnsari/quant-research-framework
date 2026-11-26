"""The pre-registered trial grid for the Pillar 4 study (PREREGISTRATION.md, sections 2 and 3).

Changing anything here after the ``pillar4-preregistration`` tag is a deviation and must be
recorded in the pre-registration's Deviations section.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product

from quant_research.backtest import rules as R

DEV_END = "2021-11-30"
HOLDOUT_START = "2021-12-01"
WARMUP_DAYS = 252
SEED = 2026

ETFS = (
    "SPY", "QQQ", "IWM", "RSP", "EFA", "VGK", "EWJ", "EEM", "EWA", "EWC", "XLE", "XLF", "XLK",
    "XLV", "XLU", "XLP", "XLI", "TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "GLD", "SLV", "GDX",
    "USO", "UNG", "DBC", "DBA", "CPER", "UUP", "FXE",
)  # fmt: skip
CURRENCIES = ("EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD", "NOK", "SEK")


def _close_rules(asset: str, short: bool) -> Iterator[R.Rule]:
    for fast, slow in product((5, 10, 20, 50), (50, 100, 200)):
        if fast < slow:
            yield R.MACrossover(asset, short, fast=fast, slow=slow)
    for n in (50, 100, 200):
        yield R.PriceVsMA(asset, short, n=n)
    for n in (20, 50, 100, 200):
        yield R.ChannelBreakout(asset, short, n=n)
    for x in (0.01, 0.02, 0.05, 0.10):
        yield R.FilterRule(asset, short, x=x)
    for lookback in (21, 63, 126, 252):
        yield R.TimeSeriesMomentum(asset, short, lookback=lookback)
    for fast, slow, sig in ((12, 26, 9), (8, 17, 9), (5, 35, 5)):
        yield R.MACD(asset, short, fast=fast, slow=slow, signal_span=sig)
    for period, (lo, hi) in product((7, 14), ((30, 70), (20, 80))):
        yield R.RSIRule(asset, short, period=period, lower=lo, upper=hi)
    for n, k in product((20, 50), (1.5, 2.0, 2.5)):
        yield R.Bollinger(asset, short, n=n, k=k)


def _ohlc_rules(asset: str) -> Iterator[R.Rule]:
    for step, max_af in ((0.02, 0.2), (0.01, 0.1), (0.04, 0.4)):
        yield R.ParabolicSAR(asset, step=step, max_af=max_af)
    for candles in (1, 2, 3):
        yield R.HeikinAshi(asset, k=candles)
    for fast, slow in ((5, 34), (10, 50)):
        yield R.AwesomeOscillator(asset, fast=fast, slow=slow)
    for ratio, hold in product((2.0, 3.0), (5, 10, 20)):
        yield R.ShootingStar(asset, ratio=ratio, hold_days=hold)
    for days, width in product((5, 10), (0.3, 0.5, 0.7)):
        yield R.DualThrust(asset, n=days, k=width)
    for n in (14, 21):
        yield R.Stochastic(asset, n=n)


def rules_for(asset: str) -> list[R.Rule]:
    """All pre-registered trials on one asset: ETFs long/flat, currencies long/short."""
    if asset in CURRENCIES:
        return list(_close_rules(asset, short=True))
    if asset in ETFS:
        return [*_close_rules(asset, short=False), *_ohlc_rules(asset)]
    raise KeyError(f"{asset} is not in the pre-registered universe")


def trial_id(rule: R.Rule) -> str:
    """Stable, readable id, e.g. 'SPY|MA crossover|fast=10,slow=50'."""
    params = ",".join(f"{k}={v:g}" for k, v in rule.params.items())
    return f"{rule.asset}|{rule.family}|{params}"
