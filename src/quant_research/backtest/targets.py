"""Checks on target weights, shared by both engines so they reject the same mistakes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.backtest.market import Market
from quant_research.errors import InvalidParameterError, MisalignedDataError, MissingValuesError


def align_targets(raw: pd.DataFrame, market: Market) -> pd.DataFrame:
    """Targets on the market's dates and assets; assets a strategy leaves out are held at 0."""
    if not raw.index.equals(market.dates):
        raise MisalignedDataError("target weights must be indexed by the market's dates")
    unknown = sorted(set(map(str, raw.columns)) - set(market.assets))
    if unknown:
        raise MisalignedDataError(f"target weights for assets not in the market: {unknown}")
    return raw.reindex(columns=market.assets, fill_value=0.0).astype(float)


def check_targets(
    weights: np.ndarray, closes: np.ndarray, dates: pd.DatetimeIndex, max_gross: float
) -> None:
    """Fail on missing weights, weights on assets with no price yet, or too much leverage.

    ``weights`` and ``closes`` have shape (T, N) (or (N,) for a single date).
    """
    w = np.atleast_2d(weights)
    c = np.atleast_2d(closes)
    if np.isnan(w).any():
        row = int(np.argwhere(np.isnan(w))[0][0])
        raise MissingValuesError(f"target weights are missing on {dates[row]:%Y-%m-%d}")
    unpriced = (w != 0) & np.isnan(c)
    if unpriced.any():
        row = int(np.argwhere(unpriced)[0][0])
        raise MissingValuesError(
            f"non-zero target weight on an asset with no price yet, {dates[row]:%Y-%m-%d}"
        )
    gross = np.abs(w).sum(axis=1)
    if (gross > max_gross + 1e-12).any():
        row = int(np.argmax(gross > max_gross + 1e-12))
        raise InvalidParameterError(
            f"gross leverage {gross[row]:.2f} on {dates[row]:%Y-%m-%d} exceeds the limit "
            f"of {max_gross:g}"
        )
