"""Known-answer tests for Pillar 5 generators and estimators."""

import numpy as np
import pytest

from quant_research.risk.var import breaches, ewma_normal_var, historical_var, kupiec_pof
from quant_research.stats.mean_reversion import ar1_fit, engle_granger, ols_levels
from quant_research.validation import overfitting as of
from quant_research.validation import synthetic as syn


def test_gbm_log_return_moments():
    lp = syn.gbm_log_prices(200_000, mu=0.08, sigma=0.2, r=syn.rng(0))
    r = np.diff(lp)
    assert r.std() * np.sqrt(252) == pytest.approx(0.2, rel=0.01)
    assert r.mean() * 252 == pytest.approx(0.08 - 0.02, abs=0.01)


def test_ou_half_life_recovered_on_long_sample():
    x = syn.ou(100_000, half_life=10, sigma=1.0, r=syn.rng(1))
    assert syn.phi_from_half_life(10) ** 10 == pytest.approx(0.5)
    assert ar1_fit(x)["half_life"] == pytest.approx(10, rel=0.05)


def test_ou_short_sample_half_life_is_biased_low():
    est = [ar1_fit(syn.ou(250, 10, 1.0, syn.rng(s)))["half_life"] for s in range(300)]
    assert np.median(est) < 10  # Kendall bias: reversion looks faster than it is


def test_garch_unconditional_variance_and_clustering():
    r, sig = syn.garch11(200_000, syn.rng(2), omega=2e-6, alpha=0.08, beta=0.9)
    assert r.var() == pytest.approx(2e-6 / 0.02, rel=0.1)
    sq = r**2
    assert np.corrcoef(sq[1:], sq[:-1])[0, 1] > 0.1  # volatility clustering


def test_cointegrated_pair_and_spurious_regression():
    y, x = syn.cointegrated_pair(2000, beta=1.5, half_life=10, r=syn.rng(3))
    eg = engle_granger(y, x)
    assert eg["beta"] == pytest.approx(1.5, abs=0.05) and eg["p_value"] < 0.01
    w = syn.independent_random_walks(2000, 2, syn.rng(4))
    lv = ols_levels(w[:, 0], w[:, 1])
    assert abs(lv["t"]) > 2  # spurious: naive t "significant" with no true relation


def test_regime_switching_matches_stationary_distribution():
    P = ((0.99, 0.01), (0.03, 0.97))
    pi = syn.stationary_distribution(P)
    assert pi == pytest.approx([0.75, 0.25])
    _, states = syn.regime_switching(200_000, syn.rng(5), transition=P)
    assert np.mean(states == 1) == pytest.approx(0.25, abs=0.02)
    stay = np.mean(states[1:][states[:-1] == 0] == 0)
    assert stay == pytest.approx(0.99, abs=0.003)


def test_kupiec_known_values():
    exact = kupiec_pof(10, 1000, 0.99)
    assert exact["lr"] == pytest.approx(0.0, abs=1e-9) and exact["p_value"] == pytest.approx(1.0)
    assert kupiec_pof(25, 1000, 0.99)["p_value"] < 0.001
    assert kupiec_pof(0, 1000, 0.99)["lr"] == pytest.approx(-2 * 1000 * np.log(0.99))


def test_var_with_true_vol_is_calibrated_on_normal_garch():
    r, sig = syn.garch11(50_000, syn.rng(6), nu=None)
    from scipy import stats

    var = stats.norm.ppf(0.99) * sig
    assert breaches(r, var).mean() == pytest.approx(0.01, abs=0.0015)
    assert not np.isnan(historical_var(r[:600])[-1])
    assert np.isnan(ewma_normal_var(r[:600])[:250]).all()


def test_expected_max_sharpe_matches_simulation():
    assert of.expected_max_sharpe(1) == 0.0
    sims = syn.rng(7).normal(0, 1, (20_000, 50)).max(axis=1)
    assert of.expected_max_sharpe(50) == pytest.approx(sims.mean(), rel=0.05)
    assert of.expected_max_sharpe(1000) > of.expected_max_sharpe(50)


def test_ma_crossover_has_no_look_ahead():
    """Plant same-day information: prices whose day-t move is the sign of a signal visible at
    close t. A look-ahead bug would earn a huge Sharpe; the correct strategy earns ~0."""
    r = syn.rng(8).normal(0, 0.01, 5000)
    lp = np.concatenate([[0.0], np.cumsum(r)])
    strat = of.ma_crossover_returns(lp, fast=1, slow=2)  # signal = sign of today's move
    assert abs(of.annualised_sharpe(strat)) < 0.6
    # Manual look-ahead version for contrast: position equals today's move sign.
    cheat = np.sign(r) * r
    assert of.annualised_sharpe(cheat) > 10
