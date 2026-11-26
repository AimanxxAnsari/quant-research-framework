"""Probability of Backtest Overfitting by combinatorially symmetric cross-validation (CSCV).

Bailey, Borwein, Lopez de Prado and Zhu (2017). Given the daily returns of N trials (T x N):

1. Cut the T rows into S equal, contiguous blocks (S even). Rows that do not fit are dropped
   from the start, so the most recent data is always used.
2. For every way of choosing S/2 blocks as the in-sample (IS) set (C(S, S/2) ways; 12,870 for
   S = 16, which the paper misprints as 12,780), the other S/2 blocks are out of sample (OOS).
3. Pick the trial with the best IS Sharpe ratio; find its rank r among the N trials OOS
   (1 = worst, N = best, ties averaged); set w = r / (N + 1) and the logit L = ln(w / (1 - w)).
4. PBO = share of splits with L <= 0: how often the IS winner does no better than the OOS
   median. About 0.5 means selection is no better than chance; near 0 means it works.

Also reported: the slope of OOS Sharpe on IS Sharpe for the selected trial (performance
degradation; negative is a bad sign) and the probability that the selected trial loses OOS.

Sharpe ratios for every split come from per-block sums and sums of squares, so each split
costs O(S x N) instead of O(T x N).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from scipy import stats

from quant_research.errors import InvalidParameterError, MissingValuesError


@dataclass(frozen=True)
class PboResult:
    pbo: float
    logits: np.ndarray  # one per split
    is_sharpe: np.ndarray  # IS Sharpe (per period) of the selected trial, per split
    oos_sharpe: np.ndarray  # OOS Sharpe (per period) of the same trial
    degradation_slope: float  # OLS slope of oos_sharpe on is_sharpe
    prob_oos_loss: float  # share of splits where the selected trial loses OOS
    n_splits: int


def _block_sums(r: np.ndarray, n_blocks: int) -> tuple[np.ndarray, np.ndarray, int]:
    size = r.shape[0] // n_blocks
    trimmed = r[r.shape[0] - size * n_blocks :]
    blocks = trimmed.reshape(n_blocks, size, r.shape[1])
    return blocks.sum(axis=1), (blocks**2).sum(axis=1), size


def _sharpe(total: np.ndarray, total_sq: np.ndarray, n: int) -> np.ndarray:
    mean = total / n
    var = (total_sq - n * mean**2) / (n - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)


def cscv(trial_returns: np.ndarray, n_blocks: int = 16) -> PboResult:
    r = np.asarray(trial_returns, dtype=float)
    if r.ndim != 2 or r.shape[1] < 2:
        raise InvalidParameterError("need a T x N matrix with at least two trials")
    if n_blocks < 2 or n_blocks % 2:
        raise InvalidParameterError("n_blocks must be an even number >= 2")
    if r.shape[0] // n_blocks < 2:
        raise InvalidParameterError("too few observations for this many blocks")
    if np.isnan(r).any():
        raise MissingValuesError("trial returns contain NaN")

    sums, sq, size = _block_sums(r, n_blocks)
    all_sum, all_sq = sums.sum(axis=0), sq.sum(axis=0)
    half = n_blocks // 2
    n = r.shape[1]
    logits, is_sr, oos_sr = [], [], []
    for chosen in combinations(range(n_blocks), half):
        idx = list(chosen)
        in_sum, in_sq = sums[idx].sum(axis=0), sq[idx].sum(axis=0)
        sr_in = _sharpe(in_sum, in_sq, size * half)
        sr_out = _sharpe(all_sum - in_sum, all_sq - in_sq, size * half)
        best = int(np.argmax(sr_in))
        w = stats.rankdata(sr_out)[best] / (n + 1)
        logits.append(np.log(w / (1 - w)))
        is_sr.append(sr_in[best])
        oos_sr.append(sr_out[best])

    lam, x, y = np.array(logits), np.array(is_sr), np.array(oos_sr)
    slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float("nan")
    return PboResult(
        pbo=float(np.mean(lam <= 0)),
        logits=lam,
        is_sharpe=x,
        oos_sharpe=y,
        degradation_slope=slope,
        prob_oos_loss=float(np.mean(y < 0)),
        n_splits=comb(n_blocks, half),
    )
