"""How selection among many backtests inflates the best Sharpe ratio.

If N strategies with no skill are tried and the best in-sample one is kept, its Sharpe ratio is
the maximum of N noisy estimates. For independent trials with Sharpe-estimate standard
deviation sd, Bailey & Lopez de Prado (2014) approximate

    E[max SR] ~ sd * ((1 - g) * Phi^{-1}(1 - 1/N) + g * Phi^{-1}(1 - 1/(N e))),

g = Euler-Mascheroni constant ~ 0.5772. This is the benchmark used by the Deflated Sharpe Ratio.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from quant_research.errors import InvalidParameterError

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sd: float = 1.0) -> float:
    if n_trials < 1:
        raise InvalidParameterError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    g = EULER_GAMMA
    return float(
        sd
        * (
            (1 - g) * stats.norm.ppf(1 - 1 / n_trials)
            + g * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    )


def annualised_sharpe(daily: np.ndarray, periods: int = 252) -> float:
    d = np.asarray(daily, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def ma_crossover_returns(log_prices: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """Daily returns of a long/short moving-average crossover.

    Signal uses closes up to t; the position is held over day t+1 (no look-ahead).
    Returns an array aligned with log returns r_1..r_{n-1}; the first ``slow`` values are 0.
    """
    p = np.asarray(log_prices, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(p)])

    def sma(w: int) -> np.ndarray:
        out = np.full(p.size, np.nan)
        out[w - 1 :] = (c[w:] - c[:-w]) / w
        return out

    signal = np.sign(sma(fast) - sma(slow))  # known at close t
    signal = np.nan_to_num(signal)
    r = np.diff(p)  # r[t-1] is the return from t-1 to t
    position = signal[:-1]  # decided at close t-1, earns r over (t-1, t]
    return np.asarray(position * r, dtype=float)
