"""The 14 rule families of the Pillar 4 study, exactly as pre-registered.

See docs/projects/pillar-4/PREREGISTRATION.md, section 3. Each rule trades one asset. At the
close of day t it outputs +1 (long 100%), 0 (flat) or -1 (short 100%, only if ``allow_short``),
using data up to that close. Where a rule says "short/flat", a long/flat rule goes flat.

Rules marked ``needs_ohlc`` use open, high or low and run on ETFs only.

Everything here is written for this repo; ideas come from the sources cited in the
pre-registration (no code copied). Every rule must pass the engine's look-ahead check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from quant_research.backtest.market import Market
from quant_research.errors import InvalidParameterError, MisalignedDataError

# ---------------------------------------------------------------- shared pieces


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def _ema(x: pd.Series, span: int) -> pd.Series:
    # recursive from the first value, so adding later data never changes earlier values
    return x.ewm(span=span, adjust=False).mean()


def _hold(events: pd.Series) -> pd.Series:
    """Carry the last signalled position forward (NaN = no new signal); flat before the first."""
    return events.ffill().fillna(0.0)


def _hysteresis(
    enter_long: np.ndarray,
    exit_long: np.ndarray,
    enter_short: np.ndarray,
    exit_short: np.ndarray,
    allow_short: bool,
) -> np.ndarray:
    """Position state machine: enter on one condition, leave on another."""
    pos = np.zeros(len(enter_long))
    state = 0.0
    for t in range(len(pos)):
        if state > 0 and exit_long[t]:
            state = 0.0
        elif state < 0 and exit_short[t]:
            state = 0.0
        if state == 0:
            if enter_long[t]:
                state = 1.0
            elif allow_short and enter_short[t]:
                state = -1.0
        pos[t] = state
    return pos


@dataclass(frozen=True)
class Rule:
    """Base class: one asset, a direction, and a ``signal`` on that asset's prices."""

    asset: str
    allow_short: bool = False
    family: ClassVar[str] = ""
    needs_ohlc: ClassVar[bool] = False

    @property
    def params(self) -> dict[str, float]:
        skip = {"asset", "allow_short"}
        return {k: v for k, v in self.__dict__.items() if k not in skip}

    @property
    def name(self) -> str:
        p = ", ".join(f"{k}={v:g}" for k, v in self.params.items())
        return f"{self.family}({p}) on {self.asset}"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def _side(self, longs: pd.Series) -> pd.Series:
        """Map a boolean 'long' signal to +1 / (-1 or 0)."""
        return longs.astype(float).where(longs, -1.0 if self.allow_short else 0.0)

    def target_weights(self, market: Market) -> pd.DataFrame:
        if self.asset not in market.assets:
            raise MisalignedDataError(f"{self.asset} is not in the market")
        bars = pd.DataFrame({"open": market.open[self.asset], "close": market.close[self.asset]})
        if self.needs_ohlc:
            if market.asset_class[self.asset] != "etf":
                raise InvalidParameterError(f"{self.family} needs high and low; ETFs only")
            bars["high"], bars["low"] = market.high[self.asset], market.low[self.asset]
        listed = bars["close"].notna()
        sig = self.signal(bars.loc[listed]).reindex(bars.index).fillna(0.0)
        if not self.allow_short:
            sig = sig.clip(lower=0.0)
        out = pd.DataFrame(0.0, index=market.dates, columns=market.assets)
        out[self.asset] = sig
        return out


# ---------------------------------------------------------------- close-based families


@dataclass(frozen=True)
class MACrossover(Rule):
    fast: int = 10
    slow: int = 50
    family: ClassVar[str] = "MA crossover"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        f, s = _sma(bars["close"], self.fast), _sma(bars["close"], self.slow)
        return self._side(f > s).where(s.notna(), 0.0)


@dataclass(frozen=True)
class PriceVsMA(Rule):
    n: int = 200
    family: ClassVar[str] = "price vs MA"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        ma = _sma(bars["close"], self.n)
        return self._side(bars["close"] > ma).where(ma.notna(), 0.0)


@dataclass(frozen=True)
class ChannelBreakout(Rule):
    n: int = 50
    family: ClassVar[str] = "channel breakout"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        c = bars["close"]
        hi = c.shift(1).rolling(self.n).max()
        lo = c.shift(1).rolling(self.n).min()
        events = pd.Series(np.nan, index=c.index)
        events[c > hi] = 1.0
        events[c < lo] = -1.0
        return _hold(events)


@dataclass(frozen=True)
class FilterRule(Rule):
    x: float = 0.05
    family: ClassVar[str] = "filter rule"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        c = bars["close"].to_numpy(dtype=float)
        pos = np.zeros(len(c))
        state, low, high = 0.0, c[0], c[0]
        for t, price in enumerate(c):
            low, high = min(low, price), max(high, price)
            if state <= 0 and price >= low * (1 + self.x):
                state, high = 1.0, price
            elif state >= 0 and price <= high * (1 - self.x):
                state, low = -1.0, price
            pos[t] = state
        return pd.Series(pos, index=bars.index)


@dataclass(frozen=True)
class TimeSeriesMomentum(Rule):
    lookback: int = 252
    family: ClassVar[str] = "time-series momentum"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        past = bars["close"] / bars["close"].shift(self.lookback) - 1.0
        return self._side(past > 0).where(past.notna(), 0.0)


@dataclass(frozen=True)
class MACD(Rule):
    fast: int = 12
    slow: int = 26
    signal_span: int = 9
    family: ClassVar[str] = "MACD"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        macd = _ema(bars["close"], self.fast) - _ema(bars["close"], self.slow)
        line = _ema(macd, self.signal_span)
        warm = np.arange(len(macd)) >= self.slow + self.signal_span
        return self._side(macd > line).where(warm, 0.0)


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-change).clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    return (100 - 100 / (1 + rs)).where(loss > 0, 100.0).where(gain.notna())


@dataclass(frozen=True)
class RSIRule(Rule):
    period: int = 14
    lower: float = 30.0
    upper: float = 70.0
    family: ClassVar[str] = "RSI"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        rsi = wilder_rsi(bars["close"], self.period).to_numpy()
        valid = ~np.isnan(rsi)
        r = np.nan_to_num(rsi, nan=50.0)
        pos = _hysteresis(
            valid & (r < self.lower), r > 50, valid & (r > self.upper), r < 50, self.allow_short
        )
        return pd.Series(pos, index=bars.index)


@dataclass(frozen=True)
class Bollinger(Rule):
    n: int = 20
    k: float = 2.0
    family: ClassVar[str] = "Bollinger bands"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        c = bars["close"]
        ma, sd = _sma(c, self.n), c.rolling(self.n).std(ddof=0)
        valid = ma.notna().to_numpy()
        cv, mv, sv = c.to_numpy(), ma.fillna(0).to_numpy(), sd.fillna(0).to_numpy()
        pos = _hysteresis(
            valid & (cv < mv - self.k * sv),
            cv > mv,
            valid & (cv > mv + self.k * sv),
            cv < mv,
            self.allow_short,
        )
        return pd.Series(pos, index=bars.index)


# ---------------------------------------------------------------- OHLC families (ETFs only)


def parabolic_sar_trend(
    high: np.ndarray, low: np.ndarray, step: float, max_af: float
) -> np.ndarray:
    """+1 in an up-trend, -1 in a down-trend (Wilder 1978); starts in an up-trend."""
    n = len(high)
    trend = np.ones(n)
    up, sar, ep, af = True, low[0], high[0], step
    for t in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, low[t - 1], low[t - 2] if t > 1 else low[t - 1])
            if low[t] < sar:
                up, sar, ep, af = False, ep, low[t], step
            elif high[t] > ep:
                ep, af = high[t], min(af + step, max_af)
        else:
            sar = max(sar, high[t - 1], high[t - 2] if t > 1 else high[t - 1])
            if high[t] > sar:
                up, sar, ep, af = True, ep, high[t], step
            elif low[t] < ep:
                ep, af = low[t], min(af + step, max_af)
        trend[t] = 1.0 if up else -1.0
    return trend


@dataclass(frozen=True)
class ParabolicSAR(Rule):
    step: float = 0.02
    max_af: float = 0.2
    family: ClassVar[str] = "Parabolic SAR"
    needs_ohlc: ClassVar[bool] = True

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        trend = parabolic_sar_trend(
            bars["high"].to_numpy(), bars["low"].to_numpy(), self.step, self.max_af
        )
        return pd.Series(trend, index=bars.index)


@dataclass(frozen=True)
class HeikinAshi(Rule):
    k: int = 2
    family: ClassVar[str] = "Heikin-Ashi"
    needs_ohlc: ClassVar[bool] = True

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        ha_close = bars[["open", "high", "low", "close"]].mean(axis=1)
        seed = (bars["open"].iloc[0] + bars["close"].iloc[0]) / 2
        ha_open = ha_close.shift(1).fillna(seed).ewm(alpha=0.5, adjust=False).mean()
        ha_open.iloc[0] = seed
        rising = (ha_close > ha_open).astype(float)
        falling = (ha_close < ha_open).astype(float)
        events = pd.Series(np.nan, index=bars.index)
        events[rising.rolling(self.k).sum() == self.k] = 1.0
        events[falling.rolling(self.k).sum() == self.k] = -1.0
        return _hold(events)


@dataclass(frozen=True)
class AwesomeOscillator(Rule):
    fast: int = 5
    slow: int = 34
    family: ClassVar[str] = "Awesome Oscillator"
    needs_ohlc: ClassVar[bool] = True

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        median = (bars["high"] + bars["low"]) / 2
        ao = _sma(median, self.fast) - _sma(median, self.slow)
        return self._side(ao > 0).where(ao.notna(), 0.0)


@dataclass(frozen=True)
class ShootingStar(Rule):
    ratio: float = 2.0
    hold_days: int = 10
    family: ClassVar[str] = "shooting star"
    needs_ohlc: ClassVar[bool] = True

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        o, h, lo, c = bars["open"], bars["high"], bars["low"], bars["close"]
        body = (c - o).abs()
        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - lo
        uptrend = (c.shift(1) > _sma(c, 20).shift(1)).fillna(False)
        star = (upper > 0) & (upper >= self.ratio * body) & (lower <= body) & uptrend
        recent = star.astype(float).rolling(self.hold_days, min_periods=1).max() > 0
        return pd.Series(np.where(recent, 0.0, 1.0), index=bars.index)


@dataclass(frozen=True)
class DualThrust(Rule):
    n: int = 5
    k: float = 0.5
    family: ClassVar[str] = "Dual Thrust (daily)"
    needs_ohlc: ClassVar[bool] = True

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        prev = bars.shift(1)
        hh, ll = prev["high"].rolling(self.n).max(), prev["low"].rolling(self.n).min()
        hc, lc = prev["close"].rolling(self.n).max(), prev["close"].rolling(self.n).min()
        rng = np.maximum(hh - lc, hc - ll)
        events = pd.Series(np.nan, index=bars.index)
        events[bars["close"] > bars["open"] + self.k * rng] = 1.0
        events[bars["close"] < bars["open"] - self.k * rng] = -1.0
        return _hold(events)


@dataclass(frozen=True)
class Stochastic(Rule):
    n: int = 14
    family: ClassVar[str] = "stochastic oscillator"
    needs_ohlc: ClassVar[bool] = True

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        hh, ll = bars["high"].rolling(self.n).max(), bars["low"].rolling(self.n).min()
        span = hh - ll
        k = (100 * (bars["close"] - ll) / span).where(span > 0, 50.0).where(hh.notna())
        d = _sma(k, 3)
        return self._side(k > d).where(d.notna(), 0.0)
