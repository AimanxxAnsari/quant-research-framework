"""Inference on Sharpe ratios: PSR, Deflated Sharpe, minimum track record, effective trials.

Units: every Sharpe ratio here is **per period** (for daily returns, per day; not annualised),
because the formulas use the number of observations. Annualise only for display.

* Probabilistic Sharpe Ratio (Bailey and Lopez de Prado 2012): the probability that the true
  Sharpe ratio exceeds a benchmark SR*, given the estimate, the sample length T and the skewness
  g3 and (non-excess) kurtosis g4 of the returns:

      PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2) )

* Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014): the PSR at the Sharpe ratio the best of
  N no-skill trials would reach by luck, SR0 = sqrt(V[SR]) x E[max of N standard normals]
  (``overfitting.expected_max_sharpe``), where V[SR] is the variance of the trials' Sharpe
  ratios. It corrects for both selection among many trials and non-normal returns.

* Minimum track record length: the T at which PSR(SR*) reaches a target probability.

* Effective number of trials: correlated trials are fewer independent bets than their count.
  Here it is the participation ratio of the trial correlation matrix's eigenvalues,
  (sum lambda)^2 / sum lambda^2 = N^2 / sum lambda^2: N for independent trials, 1 for identical
  ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from quant_research.errors import EmptySeriesError, InvalidParameterError, MissingValuesError
from quant_research.validation.overfitting import expected_max_sharpe


@dataclass(frozen=True)
class SharpeStats:
    """Per-period Sharpe ratio with the moments PSR needs (kurtosis is not excess: normal = 3)."""

    sharpe: float
    skew: float
    kurtosis: float
    n_obs: int


def sharpe_stats(returns: np.ndarray) -> SharpeStats:
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 3:
        raise EmptySeriesError("need a one-dimensional series of at least 3 returns")
    if np.isnan(r).any():
        raise MissingValuesError("returns contain NaN")
    sd = r.std(ddof=1)
    if sd == 0:
        raise InvalidParameterError("returns have zero variance; the Sharpe ratio is undefined")
    return SharpeStats(
        sharpe=float(r.mean() / sd),
        skew=float(stats.skew(r, bias=False)),
        kurtosis=float(stats.kurtosis(r, fisher=False, bias=False)),
        n_obs=int(r.size),
    )


def _sr_std_factor(sr: float, skew: float, kurtosis: float) -> float:
    var = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if not var > 0:
        raise InvalidParameterError("skew and kurtosis imply a non-positive Sharpe variance")
    return float(np.sqrt(var))


def probabilistic_sharpe(
    sr: float, n_obs: int, skew: float, kurtosis: float, sr_benchmark: float = 0.0
) -> float:
    """P(true per-period Sharpe > sr_benchmark)."""
    if n_obs < 2:
        raise InvalidParameterError("need at least 2 observations")
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1) / _sr_std_factor(sr, skew, kurtosis)
    return float(stats.norm.cdf(z))


@dataclass(frozen=True)
class DeflatedSharpe:
    dsr: float  # probability that the true Sharpe exceeds the luck threshold
    threshold: float  # SR0: expected best per-period Sharpe of n_trials no-skill trials


def deflated_sharpe(
    sr: float, n_obs: int, skew: float, kurtosis: float, sr_variance: float, n_trials: float
) -> DeflatedSharpe:
    """Deflated Sharpe Ratio of the selected trial (all Sharpe inputs per period).

    ``sr_variance`` is the variance of the per-period Sharpe ratios across all trials;
    ``n_trials`` may be non-integer (an effective number of independent trials).
    """
    if sr_variance < 0:
        raise InvalidParameterError("sr_variance must be non-negative")
    if n_trials < 1:
        raise InvalidParameterError("n_trials must be >= 1")
    threshold = expected_max_sharpe(n_trials, float(np.sqrt(sr_variance)))
    return DeflatedSharpe(
        dsr=probabilistic_sharpe(sr, n_obs, skew, kurtosis, threshold), threshold=threshold
    )


def min_track_record_length(
    sr: float, skew: float, kurtosis: float, sr_benchmark: float = 0.0, prob: float = 0.95
) -> float:
    """Observations needed for PSR(sr_benchmark) to reach ``prob`` (infinite if sr <= benchmark)."""
    if not 0 < prob < 1:
        raise InvalidParameterError("prob must be in (0, 1)")
    if sr <= sr_benchmark:
        return float("inf")
    z = stats.norm.ppf(prob)
    return float(1.0 + (_sr_std_factor(sr, skew, kurtosis) * z / (sr - sr_benchmark)) ** 2)


def effective_number_of_trials(trial_returns: np.ndarray) -> float:
    """Participation ratio of the eigenvalues of the trials' correlation matrix (T x N input)."""
    r = np.asarray(trial_returns, dtype=float)
    if r.ndim != 2 or r.shape[1] < 1:
        raise InvalidParameterError("need a T x N matrix of trial returns")
    if r.shape[1] == 1:
        return 1.0
    if np.isnan(r).any():
        raise MissingValuesError("trial returns contain NaN")
    if (r.std(axis=0) == 0).any():
        raise InvalidParameterError("a trial has constant returns; its correlation is undefined")
    eig = np.linalg.eigvalsh(np.corrcoef(r, rowvar=False))
    return float(r.shape[1] ** 2 / np.sum(eig**2))
