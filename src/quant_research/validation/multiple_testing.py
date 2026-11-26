"""Multiple-testing corrections for Sharpe ratios.

Haircut Sharpe ratios (Harvey and Liu 2015). A Sharpe ratio SR (annualised) over Y years has
t-statistic t = SR x sqrt(Y) and a two-sided single-test p-value p_S. Having tried N strategies,
the right p-value is a multiple-testing one, p_M. Converting p_M back to a t-statistic and a
Sharpe ratio gives the haircut Sharpe ratio; the haircut is 1 - SR_haircut / SR.

* ``independent``: p_M = 1 - (1 - p_S)^N (exact for independent tests);
* ``bonferroni``: p_M = min(N p_S, 1);
* with the p-values of every trial actually run: Holm (family-wise error rate) or BHY
  (Benjamini-Hochberg-Yekutieli, false discovery rate under any dependence).

The paper's further adjustment for correlated tests relies on a simulation model fitted to
published factors and is not reproduced here; with all trials logged, Holm and BHY on their
actual p-values serve that purpose.

Data snooping tests (White 2000; Hansen 2005; Romano and Wolf 2005), from the ``arch`` package:
SPA asks whether any trial beats a benchmark after allowing for the search; StepM lists the
trials that do. Both use a stationary bootstrap, so they allow for serial dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from arch.bootstrap import SPA, StepM
from scipy import stats

from quant_research.errors import InvalidParameterError, MissingValuesError


def sharpe_pvalue(sr_annual: float, years: float) -> float:
    """Two-sided p-value of an annualised Sharpe ratio over ``years`` (t = SR sqrt(years))."""
    if years <= 0:
        raise InvalidParameterError("years must be positive")
    return float(2 * stats.norm.sf(abs(sr_annual) * np.sqrt(years)))


def sharpe_from_pvalue(p: float, years: float, sign: float = 1.0) -> float:
    """Annualised Sharpe ratio whose two-sided p-value over ``years`` is ``p``."""
    if p >= 1:
        return 0.0
    return float(np.sign(sign) * stats.norm.isf(p / 2) / np.sqrt(years))


@dataclass(frozen=True)
class Haircut:
    p_single: float
    p_adjusted: float
    sharpe: float  # haircut Sharpe ratio, annualised
    haircut: float  # 1 - sharpe / original


def haircut_sharpe(
    sr_annual: float,
    years: float,
    n_tests: int,
    method: Literal["independent", "bonferroni"] = "independent",
) -> Haircut:
    if n_tests < 1:
        raise InvalidParameterError("n_tests must be >= 1")
    p = sharpe_pvalue(sr_annual, years)
    if method == "independent":
        pm = float(1 - (1 - p) ** n_tests)
    elif method == "bonferroni":
        pm = min(n_tests * p, 1.0)
    else:
        raise InvalidParameterError(f"unknown method {method!r}")
    sr = sharpe_from_pvalue(pm, years, sr_annual)
    cut = 1 - sr / sr_annual if sr_annual != 0 else 0.0
    return Haircut(p_single=p, p_adjusted=pm, sharpe=sr, haircut=float(cut))


def _pvalues(p: np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise InvalidParameterError("need a non-empty list of p-values")
    if np.isnan(arr).any() or (arr < 0).any() or (arr > 1).any():
        raise MissingValuesError("p-values must be in [0, 1] with no NaN")
    return arr


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = _pvalues(pvalues)
    m = p.size
    order = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def bhy(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg-Yekutieli adjusted p-values (false discovery rate, any dependence)."""
    p = _pvalues(pvalues)
    m = p.size
    c = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)
    raw = p[order] * m * c / np.arange(1, m + 1)
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def haircut_trials(
    sr_annual: np.ndarray, years: float, method: Literal["holm", "bhy"] = "bhy"
) -> np.ndarray:
    """Haircut Sharpe ratio of every trial, from the adjusted p-values of all trials run."""
    sr = np.asarray(sr_annual, dtype=float)
    p = np.array([sharpe_pvalue(s, years) for s in sr])
    adj = holm(p) if method == "holm" else bhy(p)
    return np.array([sharpe_from_pvalue(a, years, s) for a, s in zip(adj, sr, strict=True)])


@dataclass(frozen=True)
class SnoopingResult:
    spa_pvalue: float  # consistent SPA p-value (Hansen's recommended one)
    spa_pvalue_lower: float
    spa_pvalue_upper: float  # White's Reality Check
    better: np.ndarray  # StepM: trials that beat the benchmark at ``size``


def data_snooping(
    trial_returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    size: float = 0.05,
    reps: int = 1000,
    block_size: int | None = None,
    seed: int = 0,
) -> SnoopingResult:
    """SPA and StepM for "does any trial beat the benchmark?" (default benchmark: zero excess)."""
    r = np.asarray(trial_returns, dtype=float)
    if r.ndim != 2 or np.isnan(r).any():
        raise MissingValuesError("need a T x N matrix of trial returns with no NaN")
    bench = np.zeros(r.shape[0]) if benchmark_returns is None else np.asarray(benchmark_returns)
    if bench.shape != (r.shape[0],):
        raise InvalidParameterError("benchmark must have one return per row")
    losses, bench_loss = -r, -bench  # arch works with losses: lower is better
    spa = SPA(bench_loss, losses, block_size=block_size, reps=reps, seed=seed)
    spa.compute()
    pv = spa.pvalues
    stepm = StepM(bench_loss, losses, size=size, block_size=block_size, reps=reps, seed=seed)
    stepm.compute()
    # with array input arch labels the trials 0..N-1, so the labels are column numbers
    better = np.array(sorted(int(cast(int, i)) for i in stepm.superior_models), dtype=np.int64)
    return SnoopingResult(
        spa_pvalue=float(pv["consistent"]),
        spa_pvalue_lower=float(pv["lower"]),
        spa_pvalue_upper=float(pv["upper"]),
        better=better,
    )
