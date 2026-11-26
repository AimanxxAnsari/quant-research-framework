"""Pick the best trial on each training set and see how it does on the test set.

Input is the daily return matrix of N trials (T x N), for example every parameter setting of a
strategy, already run through the backtester. Selection is by in-sample Sharpe ratio. The
output is what an honest researcher would have earned by repeating "pick the best so far" on
data they had not used:

* walk-forward: one out-of-sample series, stitched from consecutive test windows;
* CPCV: phi complete out-of-sample paths, giving a distribution of Sharpe ratios instead of one
  number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from quant_research.errors import InvalidParameterError, MissingValuesError
from quant_research.validation.splits import Split, cpcv, cpcv_paths


def column_sharpe(returns: np.ndarray) -> np.ndarray:
    """Per-period Sharpe ratio of each column; 0 where a column has no variance."""
    sd = returns.std(axis=0, ddof=1)
    mean = returns.mean(axis=0)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)


def _check(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[1] < 1:
        raise InvalidParameterError("need a T x N matrix of trial returns")
    if np.isnan(r).any():
        raise MissingValuesError("trial returns contain NaN")
    return r


@dataclass(frozen=True)
class Selection:
    chosen: np.ndarray  # trial picked in each split
    oos: np.ndarray  # walk-forward: (T,) with NaN outside test rows; CPCV: (n_paths, T)


def select_out_of_sample(returns: np.ndarray, splits: Sequence[Split]) -> Selection:
    """Stitch the test returns of each split's in-sample winner into one series.

    Test sets must not overlap (walk-forward or k-fold). Rows never tested stay NaN.
    """
    r = _check(returns)
    out = np.full(r.shape[0], np.nan)
    chosen = []
    for s in splits:
        if not np.isnan(out[s.test]).all():
            raise InvalidParameterError("test sets overlap; use cpcv_out_of_sample for CPCV")
        best = int(np.argmax(column_sharpe(r[s.train])))
        out[s.test] = r[s.test, best]
        chosen.append(best)
    return Selection(np.array(chosen), out)


def cpcv_out_of_sample(
    returns: np.ndarray,
    n_groups: int,
    n_test_groups: int,
    label_end: np.ndarray | None = None,
    embargo: int = 0,
) -> Selection:
    """CPCV: for each split pick the in-sample winner; assemble its test returns into paths."""
    r = _check(returns)
    splits = cpcv(r.shape[0], n_groups, n_test_groups, label_end, embargo)
    chosen = np.array([int(np.argmax(column_sharpe(r[s.train]))) for s in splits])
    groups = np.array_split(np.arange(r.shape[0]), n_groups)
    plan = cpcv_paths(n_groups, n_test_groups)
    paths = np.empty((plan.shape[0], r.shape[0]))
    for p, row in enumerate(plan):
        for g, split_id in enumerate(row):
            paths[p, groups[g]] = r[groups[g], chosen[split_id]]
    return Selection(chosen, paths)
