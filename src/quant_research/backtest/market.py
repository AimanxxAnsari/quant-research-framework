"""Market data for a backtest: prices, interest rates and asset metadata on one calendar.

Conventions (ADR-0009):

* Prices are in USD per unit. ETFs use split- and dividend-adjusted open and close, so price
  changes include dividends (total return). FX assets are a foreign-currency balance priced in
  USD per unit of that currency; FX has one daily fixing, used as both open and close.
* ``asset_rate`` (percent per annum, simple) is earned by a balance of the asset held from one
  date to the next, scaled by calendar days / ``asset_day_count``. It is zero for ETFs and the
  foreign short rate for FX. Negative balances pay the same rate.
* ``cash_rate`` is the USD short rate earned (or paid, if negative) on USD cash the same way.
* An asset may start trading after the first date (leading missing prices), but once it has a
  price it must have one on every later date.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass

import numpy as np
import pandas as pd

from quant_research.backtest.config import ASSET_CLASSES
from quant_research.errors import (
    ConfigError,
    EmptySeriesError,
    InvalidParameterError,
    MisalignedDataError,
    MissingValuesError,
)

DAY_COUNTS = (360, 365)


@dataclass(frozen=True)
class Market:
    open: pd.DataFrame
    close: pd.DataFrame
    asset_class: Mapping[str, str]
    asset_rate: pd.DataFrame
    asset_day_count: Mapping[str, int]
    cash_rate: pd.Series
    cash_day_count: int = 360
    validate: InitVar[bool] = True

    def __post_init__(self, validate: bool) -> None:
        if validate:
            self._check_shape()
            self._check_prices()
            self._check_rates()

    def _check_shape(self) -> None:
        idx = self.close.index
        if len(idx) < 2:
            raise EmptySeriesError("a market needs at least two dates")
        if not isinstance(idx, pd.DatetimeIndex) or not idx.is_monotonic_increasing:
            raise MisalignedDataError("dates must be an increasing DatetimeIndex")
        if not idx.is_unique:
            raise MisalignedDataError("dates must be unique")
        for name, frame in (("open", self.open), ("asset_rate", self.asset_rate)):
            if not frame.index.equals(idx) or list(frame.columns) != list(self.close.columns):
                raise MisalignedDataError(f"{name} must have the same dates and assets as close")
        if not self.cash_rate.index.equals(idx):
            raise MisalignedDataError("cash_rate must have the same dates as close")
        assets = [str(c) for c in self.close.columns]
        if set(self.asset_class) != set(assets) or set(self.asset_day_count) != set(assets):
            raise ConfigError("asset_class and asset_day_count must cover exactly the assets")
        bad = {a: c for a, c in self.asset_class.items() if c not in ASSET_CLASSES}
        if bad:
            raise ConfigError(f"unknown asset classes {bad}")
        counts = [*self.asset_day_count.values(), self.cash_day_count]
        if any(d not in DAY_COUNTS for d in counts):
            raise ConfigError(f"day counts must be one of {DAY_COUNTS}")

    def _check_prices(self) -> None:
        o, c = self.open.to_numpy(dtype=float), self.close.to_numpy(dtype=float)
        if not np.array_equal(np.isnan(o), np.isnan(c)):
            raise MissingValuesError("open and close must be missing on the same dates")
        listed = ~np.isnan(c)
        started = np.maximum.accumulate(listed, axis=0)
        if (started & ~listed).any():
            col = int(np.argwhere(started & ~listed)[0][1])
            raise MissingValuesError(
                f"{self.close.columns[col]} has a missing price after it started trading"
            )
        if (c[listed] <= 0).any() or (o[listed] <= 0).any():
            raise InvalidParameterError("prices must be positive")

    def _check_rates(self) -> None:
        if self.asset_rate.isna().any().any() or self.cash_rate.isna().any():
            raise MissingValuesError("interest rates must have no missing values")
        etfs = [a for a, c in self.asset_class.items() if c == "etf"]
        if etfs and (self.asset_rate[etfs].to_numpy() != 0).any():
            raise InvalidParameterError("ETFs earn no interest of their own; asset_rate must be 0")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def assets(self) -> list[str]:
        return [str(c) for c in self.close.columns]

    def up_to(self, date: pd.Timestamp) -> Market:
        """The market as it was known at the close of ``date`` (all rows up to and including it)."""
        end = int(self.dates.searchsorted(date, side="right"))
        return Market(
            open=self.open.iloc[:end],
            close=self.close.iloc[:end],
            asset_class=self.asset_class,
            asset_rate=self.asset_rate.iloc[:end],
            asset_day_count=self.asset_day_count,
            cash_rate=self.cash_rate.iloc[:end],
            cash_day_count=self.cash_day_count,
            validate=False,  # a slice of a valid market is valid
        )

    def accrual_factors(self) -> tuple[np.ndarray, np.ndarray]:
        """Growth of USD cash and of each asset balance from date t-1 to date t from interest.

        Returns (cash, assets) with shapes (T,) and (T, N); row 0 is 1. The rate in force on
        date t-1 applies to the calendar days until date t.
        """
        days = np.diff(self.dates.to_numpy()).astype("timedelta64[D]").astype(float)
        cash = np.ones(len(self.dates))
        cash[1:] = (
            1.0 + self.cash_rate.to_numpy(dtype=float)[:-1] / 100 * days / self.cash_day_count
        )
        basis = np.array([self.asset_day_count[a] for a in self.assets], dtype=float)
        rates = self.asset_rate.to_numpy(dtype=float)
        assets = np.ones_like(rates)
        assets[1:] = 1.0 + rates[:-1] / 100 * days[:, None] / basis[None, :]
        return cash, assets
