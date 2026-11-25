"""Generators for data whose true parameters are known.

A method that cannot recover the truth on synthetic data cannot be trusted on real data.
Every generator takes a ``numpy.random.Generator`` so experiments are reproducible.
"""

from __future__ import annotations

import numpy as np


def rng(seed: int | None = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def gbm_log_prices(
    n: int, mu: float, sigma: float, r: np.random.Generator, dt: float = 1 / 252, s0: float = 100.0
) -> np.ndarray:
    """Exact GBM sampling: log returns ~ N((mu - sigma^2/2) dt, sigma^2 dt)."""
    steps = r.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    return np.log(s0) + np.concatenate([[0.0], np.cumsum(steps)])


def phi_from_half_life(half_life: float) -> float:
    """AR(1) coefficient whose deviations halve every ``half_life`` steps: phi^h = 1/2."""
    return 0.5 ** (1.0 / half_life)


def ou(
    n: int,
    half_life: float,
    sigma: float,
    r: np.random.Generator,
    mean: float = 0.0,
    x0: float | None = None,
) -> np.ndarray:
    """Ornstein-Uhlenbeck sampled exactly at unit spacing, i.e. a stationary AR(1):
    x_t = mean + phi (x_{t-1} - mean) + sigma * e_t. Starts from the stationary distribution."""
    phi = phi_from_half_life(half_life)
    stat_sd = sigma / np.sqrt(1 - phi**2)
    x = np.empty(n)
    x[0] = mean + (r.normal(0, stat_sd) if x0 is None else x0)
    e = r.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = mean + phi * (x[t - 1] - mean) + e[t]
    return x


def garch11(
    n: int,
    r: np.random.Generator,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    nu: float | None = 5.0,
    burn: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """GARCH(1,1) returns and their true conditional volatility.

    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}; innovations standardised
    Student-t with ``nu`` degrees of freedom (unit variance), or normal if nu is None.
    Unconditional variance omega / (1 - alpha - beta).
    """
    if alpha + beta >= 1:
        raise ValueError("need alpha + beta < 1 for a stationary process")
    total = n + burn
    if nu is None:
        z = r.normal(0, 1, total)
    else:
        z = r.standard_t(nu, total) * np.sqrt((nu - 2) / nu)
    s2 = np.empty(total)
    ret = np.empty(total)
    s2[0] = omega / (1 - alpha - beta)
    ret[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, total):
        s2[t] = omega + alpha * ret[t - 1] ** 2 + beta * s2[t - 1]
        ret[t] = np.sqrt(s2[t]) * z[t]
    return ret[burn:], np.sqrt(s2[burn:])


def cointegrated_pair(
    n: int,
    beta: float,
    half_life: float,
    r: np.random.Generator,
    spread_sigma: float = 0.01,
    x_sigma: float = 0.01,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Log prices (y, x) with y = alpha + beta * x + spread, x a random walk and the spread
    a mean-reverting OU process with the given half-life. True hedge ratio = beta."""
    x = np.cumsum(r.normal(0, x_sigma, n))
    spread = ou(n, half_life, spread_sigma, r)
    return alpha + beta * x + spread, x


def independent_random_walks(
    n: int, k: int, r: np.random.Generator, sigma: float = 0.01
) -> np.ndarray:
    """k independent random walks of length n (columns): no true relationship exists."""
    return np.cumsum(r.normal(0, sigma, (n, k)), axis=0)


def stationary_distribution(transition) -> np.ndarray:
    """pi with pi P = pi: the long-run share of time spent in each state."""
    P = np.asarray(transition, dtype=float)
    evals, evecs = np.linalg.eig(P.T)
    pi = np.real(evecs[:, np.argmin(np.abs(evals - 1))])
    return pi / pi.sum()


def regime_switching(
    n: int,
    r: np.random.Generator,
    means=(0.0005, -0.001),
    vols=(0.007, 0.02),
    transition=((0.99, 0.01), (0.03, 0.97)),
) -> tuple[np.ndarray, np.ndarray]:
    """Returns from a hidden Markov chain with state-dependent mean and volatility.
    Row i of ``transition`` holds the probabilities of moving from state i to each state."""
    P = np.asarray(transition, dtype=float)
    cum = np.cumsum(P, axis=1)
    states = np.empty(n, dtype=int)
    states[0] = r.choice(len(P), p=stationary_distribution(P))
    u = r.random(n)
    for t in range(1, n):
        states[t] = int(np.searchsorted(cum[states[t - 1]], u[t], side="right"))
    m, s = np.asarray(means), np.asarray(vols)
    return m[states] + s[states] * r.normal(0, 1, n), states
