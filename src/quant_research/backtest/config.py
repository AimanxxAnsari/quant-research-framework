"""Backtest settings: fill timing, transaction costs and safety limits (ADR-0009).

Loaded from ``config/backtest.yaml``; every field is validated, so a typo fails loudly instead of
quietly running with a default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np
import yaml

from quant_research.data.settings import Settings
from quant_research.errors import ConfigError, InvalidParameterError

FillRule = Literal["next_open", "next_close"]
FILL_RULES: tuple[FillRule, ...] = ("next_open", "next_close")
ASSET_CLASSES = ("etf", "fx")


@dataclass(frozen=True)
class CostModel:
    """Fixed one-way cost per asset class, in basis points of traded notional.

    A trade of notional N in an asset of class c costs N * bps[c] * multiplier / 10,000, paid
    from USD cash. ``multiplier`` exists for the 1x / 2x / 3x cost-sensitivity sweep.
    """

    bps: Mapping[str, float] = field(default_factory=lambda: {"etf": 2.0, "fx": 1.0})
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        unknown = sorted(set(self.bps) - set(ASSET_CLASSES))
        if unknown:
            raise ConfigError(f"cost bps given for unknown asset classes {unknown}")
        if any(v < 0 for v in self.bps.values()) or self.multiplier < 0:
            raise InvalidParameterError("costs must be non-negative")

    def fractions(self, asset_classes: Sequence[str]) -> np.ndarray:
        """Cost per unit of traded notional for each asset, as a fraction (not bps)."""
        missing = sorted(set(asset_classes) - set(self.bps))
        if missing:
            raise ConfigError(f"no cost assumption for asset classes {missing}")
        return np.array([self.bps[c] * self.multiplier / 1e4 for c in asset_classes], dtype=float)

    def scaled(self, multiplier: float) -> CostModel:
        return replace(self, multiplier=multiplier)


@dataclass(frozen=True)
class BacktestConfig:
    """How a backtest turns target weights into positions.

    fill: a decision made at the close of day t is filled at the open of day t+1 (``next_open``)
        or at the close of day t+1 (``next_close``). Never at the close of day t.
    initial_nav: starting capital in USD, all in cash.
    costs: transaction-cost model.
    max_gross_leverage: upper bound on sum(|weights|); larger targets raise an error.
    lookahead_checks: number of dates on which the weights engine recomputes a strategy on
        truncated data to catch look-ahead (0 switches the check off).
    """

    fill: FillRule = "next_open"
    initial_nav: float = 1_000_000.0
    costs: CostModel = field(default_factory=CostModel)
    max_gross_leverage: float = 4.0
    lookahead_checks: int = 20

    def __post_init__(self) -> None:
        if self.fill not in FILL_RULES:
            raise ConfigError(f"fill must be one of {FILL_RULES}, got {self.fill!r}")
        if not self.initial_nav > 0:
            raise InvalidParameterError("initial_nav must be positive")
        if not self.max_gross_leverage > 0:
            raise InvalidParameterError("max_gross_leverage must be positive")
        if self.lookahead_checks < 0:
            raise InvalidParameterError("lookahead_checks must be >= 0")

    def with_fill(self, fill: FillRule) -> BacktestConfig:
        return replace(self, fill=fill)


def config_from_mapping(raw: Mapping[str, Any]) -> BacktestConfig:
    known = {
        "fill",
        "initial_nav",
        "costs_bps",
        "cost_multiplier",
        "max_gross_leverage",
        "lookahead_checks",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown backtest settings {unknown}")
    costs = CostModel(
        bps={str(k): float(v) for k, v in (raw.get("costs_bps") or {}).items()},
        multiplier=float(raw.get("cost_multiplier", 1.0)),
    )
    return BacktestConfig(
        fill=cast(FillRule, raw.get("fill", "next_open")),
        initial_nav=float(raw.get("initial_nav", 1_000_000.0)),
        costs=costs,
        max_gross_leverage=float(raw.get("max_gross_leverage", 4.0)),
        lookahead_checks=int(raw.get("lookahead_checks", 20)),
    )


def load_config(settings: Settings) -> BacktestConfig:
    raw = yaml.safe_load((settings.config_dir / "backtest.yaml").read_text()) or {}
    return config_from_mapping(raw)
