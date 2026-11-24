"""Known-answer tests for the Pillar 2 statistics toolkit."""

import numpy as np
import pandas as pd
import pytest

from quant_research.stats import dependence as dep
from quant_research.stats import distribution
from quant_research.stats import random_walk as rw
from quant_research.stats import returns as ret
from quant_research.stats import volatility as vol

DAYS = pd.bdate_range("2020-01-01", periods=2000)


def gbm_bars(n_days=2000, sigma_annual=0.20, steps=390, seed=0, gap_sigma=0.0):
    """Daily OHLC from a zero-drift GBM simulated on an intraday grid (known volatility)."""
    rng = np.random.default_rng(seed)
    s = sigma_annual / np.sqrt(252 * steps)
    log_p = 0.0
    rows = []
    for _ in range(n_days):
        open_ = log_p + (rng.normal(0, gap_sigma) if gap_sigma else 0.0)
        path = open_ + np.cumsum(rng.normal(0, s, steps))
        path = np.concatenate([[open_], path])
        rows.append(np.exp([path[0], path.max(), path.min(), path[-1]]))
        log_p = path[-1]
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=DAYS[:n_days])


# ------------------------------------------------------------------ returns


def test_log_vs_simple_and_cagr():
    p = pd.Series([100.0, 110.0, 99.0], index=DAYS[:3])
    assert ret.simple_returns(p).tolist() == pytest.approx([0.10, -0.10])
    assert ret.log_returns(p).sum() == pytest.approx(np.log(0.99))
    assert ret.cumulative_return(ret.simple_returns(p)) == pytest.approx(-0.01)
    assert ret.cagr(pd.Series([0.01] * 252)) == pytest.approx(1.01**252 - 1)


def test_max_drawdown_hand_computed():
    p = pd.Series([100, 120, 90, 130, 104, 131], index=DAYS[:6], dtype=float)
    m = ret.max_drawdown(p)
    assert m["max_drawdown"] == pytest.approx(-0.25)  # 120 -> 90
    assert (m["peak"], m["trough"], m["recovery"]) == (DAYS[1], DAYS[2], DAYS[3])


# ------------------------------------------------------------------ volatility


def test_ewma_hand_computed():
    r = pd.Series([0.0, 0.01, -0.02, 0.03], index=DAYS[:4])
    out = vol.ewma_variance(r, lam=0.9, init_window=2)
    s2 = np.var([0.0, 0.01], ddof=1)
    assert out.iloc[2] == pytest.approx(s2)  # forecast for t=2 uses data up to t=1
    assert out.iloc[3] == pytest.approx(0.9 * s2 + 0.1 * 0.02**2)


def test_range_estimators_recover_true_volatility():
    bars = gbm_bars(sigma_annual=0.20)
    est = vol.all_estimators(bars, window=21).drop(columns="ewma").dropna()
    means = est.mean()
    # Discrete sampling biases range estimators slightly low; all should be close to 20%.
    assert means.between(0.185, 0.21).all(), means


def test_range_estimators_are_more_efficient_than_close_to_close():
    est = vol.all_estimators(gbm_bars(sigma_annual=0.20), window=21).dropna()
    noise = (est - 0.20).std()
    assert noise["parkinson"] < noise["close_to_close"]
    assert noise["garman_klass"] < noise["parkinson"]


def test_yang_zhang_captures_overnight_gaps_others_miss():
    bars = gbm_bars(sigma_annual=0.20, gap_sigma=0.01)  # 1% overnight gaps (~16% annual)
    est = vol.all_estimators(bars, window=21).dropna().mean()
    true_total = np.sqrt(0.20**2 + 0.01**2 * 252)
    assert est["yang_zhang"] == pytest.approx(true_total, rel=0.05)
    assert est["garman_klass"] < 0.9 * true_total  # intraday-only estimator misses the gaps


# ------------------------------------------------------------------ distribution


def test_describe_flags_fat_tails():
    rng = np.random.default_rng(0)
    normal = distribution.describe(pd.Series(rng.normal(0, 0.01, 5000)))
    fat = distribution.describe(pd.Series(rng.standard_t(3, 5000) * 0.01))
    assert abs(normal["excess_kurtosis"]) < 0.3 and normal["jb_pvalue"] > 0.01
    assert fat["excess_kurtosis"] > 2 and fat["jb_pvalue"] < 1e-6


# ------------------------------------------------------------------ dependence


def test_beta_recovers_known_slope():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0, 1, 2000))
    y = 2.0 * x + rng.normal(0, 0.5, 2000)
    b = dep.beta(y, x)
    assert b["beta"] == pytest.approx(2.0, abs=3 * b["se"])
    assert b["r2"] == pytest.approx(4 / 4.25, abs=0.02)


def test_lead_lag_detects_delayed_reaction():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(0, 1, 3000))
    y = 0.8 * x.shift(1) + rng.normal(0, 0.6, 3000)  # y reacts one day later
    ll = dep.lead_lag_corr(x, y, range(-2, 3))
    assert ll.idxmax() == 1 and ll[1] > 0.7 and abs(ll[0]) < 0.06


def test_nonsynchronous_sampling_biases_daily_corr_and_weekly_recovers_it():
    """Two assets driven by one factor; y is sampled 'later in the day' so half of each day's
    factor move lands in the next observation."""
    rng = np.random.default_rng(3)
    n = 5000
    f = rng.normal(0, 1, 2 * n)  # half-day factor moves
    x = f[0::2] + f[1::2]  # sampled at end of day
    y = f[1::2] + np.r_[f[2::2], 0.0]  # sampled mid-way: second half of today + first of next
    x, y = (
        pd.Series(x, index=pd.bdate_range("2000-01-03", periods=n)),
        pd.Series(y, index=pd.bdate_range("2000-01-03", periods=n)),
    )
    daily = x.corr(y)
    wx = x.resample("W-WED").sum()
    wy = y.resample("W-WED").sum()
    assert daily == pytest.approx(0.5, abs=0.05)
    assert wx.corr(wy) > 0.85
    assert dep.sum_of_lags_corr(x, y, 1) == pytest.approx(1.0, abs=0.06)


# ------------------------------------------------------------------ random walk


def test_variance_ratio_random_walk_and_ar1():
    rng = np.random.default_rng(4)
    iid = rng.normal(0, 1, 5000)
    v = rw.variance_ratio(iid, 2)
    assert v.vr == pytest.approx(1.0, abs=0.05) and abs(v.z_star) < 2.5
    e = rng.normal(0, 1, 20000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for t in range(1, e.size):
        ar[t] = 0.3 * ar[t - 1] + e[t]
    assert rw.variance_ratio(ar, 2).vr == pytest.approx(1.3, abs=0.03)  # VR(2) = 1 + rho_1
    assert rw.variance_ratio(ar, 2).z_star > 10
    neg = rw.variance_ratio(-np.diff(e), 2)  # MA(1) with rho_1 = -0.5
    assert neg.vr == pytest.approx(0.5, abs=0.03)


def test_hurst_random_walk_near_shuffled_baseline():
    x = np.random.default_rng(5).normal(0, 1, 4000)
    h = rw.hurst_rs(x)
    base, sd = rw.hurst_shuffled_baseline(x, n_shuffles=10)
    assert 0.45 < h < 0.65  # small-sample upward bias is expected
    assert abs(h - base) < 4 * sd + 0.03


def test_lo_modified_rs_does_not_mistake_short_memory_for_long_memory():
    rng = np.random.default_rng(6)
    e = rng.normal(0, 1, 5000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for t in range(1, e.size):
        ar[t] = 0.7 * ar[t - 1] + e[t]  # short memory only
    base, sd = rw.hurst_shuffled_baseline(ar, n_shuffles=10)
    assert rw.hurst_rs(ar) > base + 3 * sd  # classical R/S is fooled by short memory
    lo = rw.lo_modified_rs(ar)
    assert not lo["reject_no_long_memory"] and lo["q"] > 0  # Lo's test is not
