"""Train / test splits for time series: walk-forward, purged k-fold and CPCV.

Positions are integer row numbers 0..T-1 in time order.

Purging and embargo (Lopez de Prado 2018, ch. 7 and 12). An observation i "uses" the rows from i
to ``label_end[i]`` (for example a return over the next h days). A training row is purged if
that interval overlaps the rows used by a test block's labels (before the block, because its
label reaches into the test period; after it, because it starts while test labels still run).
The embargo then drops the next ``embargo`` rows after the purged ones, since serial
correlation can leak test information into them. With daily strategy returns and no
overlapping labels, ``label_end[i] = i`` and only the embargo matters.

Combinatorial purged cross-validation (CPCV): cut the rows into G groups and use every choice
of k groups as the test set: C(G, k) splits. Each group is tested C(G-1, k-1) times, so the test
pieces can be assembled into phi = k / G x C(G, k) complete backtest paths, each covering every
row once.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from quant_research.errors import InvalidParameterError


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray
    test_groups: tuple[int, ...] = ()


def _label_end(n_obs: int, label_end: np.ndarray | None) -> np.ndarray:
    if label_end is None:
        return np.arange(n_obs)
    end = np.asarray(label_end, dtype=int)
    if end.shape != (n_obs,) or (end < np.arange(n_obs)).any() or (end >= n_obs).any():
        raise InvalidParameterError("label_end must give, for each row, a row at or after it")
    return end


def _contiguous_blocks(idx: np.ndarray) -> list[tuple[int, int]]:
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def purged_train(
    n_obs: int, test: np.ndarray, label_end: np.ndarray | None = None, embargo: int = 0
) -> np.ndarray:
    """All rows not in ``test``, minus purged and embargoed rows."""
    if embargo < 0:
        raise InvalidParameterError("embargo must be >= 0")
    end = _label_end(n_obs, label_end)
    keep = np.ones(n_obs, dtype=bool)
    keep[test] = False
    rows = np.arange(n_obs)
    for a, b in _contiguous_blocks(np.sort(test)):
        span_end = int(end[a : b + 1].max())
        keep &= ~((rows <= span_end) & (end >= a))  # label overlaps the test block
        keep[span_end + 1 : span_end + 1 + embargo] = False  # after the purged rows
    return rows[keep]


def _groups(n_obs: int, n_groups: int) -> list[np.ndarray]:
    if not 2 <= n_groups <= n_obs:
        raise InvalidParameterError("need 2 <= n_groups <= number of observations")
    return np.array_split(np.arange(n_obs), n_groups)


def cpcv(
    n_obs: int,
    n_groups: int,
    n_test_groups: int,
    label_end: np.ndarray | None = None,
    embargo: int = 0,
) -> list[Split]:
    """Every choice of ``n_test_groups`` of ``n_groups`` groups as the test set, purged."""
    if not 1 <= n_test_groups < n_groups:
        raise InvalidParameterError("need 1 <= n_test_groups < n_groups")
    groups = _groups(n_obs, n_groups)
    out = []
    for chosen in combinations(range(n_groups), n_test_groups):
        test = np.concatenate([groups[g] for g in chosen])
        out.append(Split(purged_train(n_obs, test, label_end, embargo), test, chosen))
    return out


def purged_kfold(
    n_obs: int, n_folds: int, label_end: np.ndarray | None = None, embargo: int = 0
) -> list[Split]:
    """k-fold with contiguous folds, purged and embargoed: CPCV with one test group."""
    return cpcv(n_obs, n_folds, 1, label_end, embargo)


def cpcv_paths(n_groups: int, n_test_groups: int) -> np.ndarray:
    """Which split supplies each group on each backtest path: array (n_paths, n_groups).

    Splits are numbered in ``cpcv`` order. The j-th split (in that order) that tests group g
    feeds path j for that group.
    """
    splits = list(combinations(range(n_groups), n_test_groups))
    per_group: list[list[int]] = [[] for _ in range(n_groups)]
    for s, chosen in enumerate(splits):
        for g in chosen:
            per_group[g].append(s)
    n_paths = len(per_group[0])
    return np.array([[per_group[g][p] for g in range(n_groups)] for p in range(n_paths)])


def walk_forward(
    n_obs: int,
    train_size: int,
    test_size: int,
    expanding: bool = True,
    embargo: int = 0,
) -> list[Split]:
    """Train on the past, test on the next ``test_size`` rows, step forward, repeat.

    Expanding: training starts at row 0 and grows. Rolling: the last ``train_size`` rows.
    ``embargo`` rows between training and test are left out of training.
    """
    if train_size < 1 or test_size < 1 or embargo < 0:
        raise InvalidParameterError("train_size and test_size must be >= 1, embargo >= 0")
    out = []
    start = train_size + embargo
    while start < n_obs:
        stop = min(start + test_size, n_obs)
        train_end = start - embargo
        train_start = 0 if expanding else max(0, train_end - train_size)
        out.append(Split(np.arange(train_start, train_end), np.arange(start, stop)))
        start = stop
    if not out:
        raise InvalidParameterError("not enough observations for one walk-forward step")
    return out
