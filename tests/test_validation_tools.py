"""Pillar 4 validation tools against published examples and known answers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from quant_research.errors import InvalidParameterError, MissingValuesError
from quant_research.validation.multiple_testing import (
    bhy,
    data_snooping,
    haircut_sharpe,
    haircut_trials,
    holm,
)
from quant_research.validation.pbo import cscv
from quant_research.validation.selection import (
    column_sharpe,
    cpcv_out_of_sample,
    select_out_of_sample,
)
from quant_research.validation.sharpe import (
    deflated_sharpe,
    effective_number_of_trials,
    min_track_record_length,
    probabilistic_sharpe,
    sharpe_stats,
)
from quant_research.validation.splits import cpcv, cpcv_paths, purged_kfold, walk_forward

# ---------------------------------------------------------------- Deflated Sharpe


def _paper_dsr(n_trials: float, skew: float = -3.0, kurt: float = 10.0):
    # Bailey and Lopez de Prado (2014), "A numerical example": annualised SR 2.5 over 5 years of
    # daily data (T = 1250, 250 a year), variance of the trials' annualised SRs 1/2.
    return deflated_sharpe(2.5 / np.sqrt(250), 1250, skew, kurt, 0.5 / 250, n_trials)


def test_deflated_sharpe_reproduces_the_paper_example():
    res = _paper_dsr(100)
    assert res.threshold == pytest.approx(0.1132, abs=5e-5)  # SR0, non-annualised
    assert res.dsr == pytest.approx(0.9004, abs=5e-5)  # "only a 90% chance"
    assert _paper_dsr(46).dsr == pytest.approx(0.9505, abs=5e-5)  # "would have been 0.9505"
    # "If the strategy had exhibited Normal returns, DSR >= 0.95 after N = 88"
    assert _paper_dsr(88, 0.0, 3.0).dsr >= 0.95 > _paper_dsr(89, 0.0, 3.0).dsr


def test_psr_for_normal_returns_and_min_track_record():
    sr, n = 0.08, 500
    # iid normal returns: Var(SR estimate) = (1 + SR^2 / 2) / (T - 1) (Lo 2002)
    assert probabilistic_sharpe(sr, n, 0.0, 3.0) == pytest.approx(
        stats.norm.cdf(sr * np.sqrt(n - 1) / np.sqrt(1 + sr**2 / 2))
    )
    trl = min_track_record_length(0.05, -1.0, 6.0, prob=0.95)
    assert probabilistic_sharpe(0.05, int(np.ceil(trl)), -1.0, 6.0) >= 0.95
    assert probabilistic_sharpe(0.05, int(np.floor(trl)), -1.0, 6.0) < 0.95
    assert min_track_record_length(0.0, 0.0, 3.0) == float("inf")
    # negative skew and fat tails need a longer record for the same Sharpe ratio
    assert trl > min_track_record_length(0.05, 0.0, 3.0)


def test_dsr_falls_as_more_trials_are_run():
    values = [_paper_dsr(n).dsr for n in (1, 10, 100, 1000)]
    assert values == sorted(values, reverse=True)


def test_effective_number_of_trials():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (5000, 3))
    independent = rng.normal(0, 1, (5000, 20))
    clustered = np.repeat(base, 4, axis=1) + rng.normal(0, 1e-6, (5000, 12))
    assert effective_number_of_trials(independent) == pytest.approx(20, rel=0.05)
    assert effective_number_of_trials(clustered) == pytest.approx(3, rel=0.02)
    assert effective_number_of_trials(np.repeat(base[:, :1], 5, axis=1)) == pytest.approx(1)


def test_sharpe_stats_and_bad_inputs():
    s = sharpe_stats(np.random.default_rng(1).normal(0.001, 0.01, 200_000))
    assert s.skew == pytest.approx(0, abs=0.02) and s.kurtosis == pytest.approx(3, abs=0.05)
    with pytest.raises(InvalidParameterError):
        sharpe_stats(np.ones(10))
    with pytest.raises(MissingValuesError):
        sharpe_stats(np.array([0.1, np.nan, 0.2]))
    with pytest.raises(InvalidParameterError):
        probabilistic_sharpe(1.0, 100, 5.0, 3.0)  # implies a negative variance
    with pytest.raises(InvalidParameterError):
        deflated_sharpe(0.1, 100, 0, 3, 0.001, 0.5)


# ---------------------------------------------------------------- PBO


def test_cscv_split_count_and_skilled_trial():
    r = np.random.default_rng(1).normal(0, 0.01, (1000, 20))
    r[:, 7] += 0.002
    res = cscv(r, 16)
    assert res.n_splits == len(res.logits) == 12_870  # the paper prints 12,780: a typo
    assert res.pbo == 0.0 and res.prob_oos_loss == 0.0


def test_pbo_on_pure_noise_averages_one_half():
    # One dataset's PBO is noisy (0.3 to 0.9) because all splits reuse the same data, so the
    # known answer is the average over many independent datasets.
    pbos = [cscv(np.random.default_rng(s).normal(0, 0.01, (800, 10)), 8).pbo for s in range(200)]
    assert np.mean(pbos) == pytest.approx(0.5, abs=0.06)


def test_cscv_bad_inputs():
    r = np.random.default_rng(0).normal(0, 1, (100, 5))
    for bad in (dict(n_blocks=7), dict(n_blocks=0)):
        with pytest.raises(InvalidParameterError):
            cscv(r, **bad)
    with pytest.raises(InvalidParameterError):
        cscv(r[:, :1])
    with pytest.raises(MissingValuesError):
        cscv(np.where(r > 2, np.nan, r))


# ---------------------------------------------------------------- splits


def test_cpcv_counts_and_paths():
    splits = cpcv(120, 6, 2)
    assert len(splits) == 15  # C(6, 2)
    plan = cpcv_paths(6, 2)
    assert plan.shape == (5, 6)  # phi = 2/6 x 15 = 5 paths, one split per group on each
    for g in range(6):
        assert all(g in splits[s].test_groups for s in plan[:, g])
    counts = np.bincount(plan.ravel(), minlength=15)
    assert (counts == 2).all()  # every split feeds k = 2 path-group slots


def test_purging_and_embargo():
    n = 100
    label_end = np.minimum(np.arange(n) + 5, n - 1)  # each label uses the next 5 rows
    folds = purged_kfold(n, 5, label_end=label_end, embargo=3)
    assert sorted(np.concatenate([f.test for f in folds]).tolist()) == list(range(n))
    middle = folds[2]  # test rows 40..59
    assert set(middle.train).isdisjoint(middle.test)
    assert set(range(35, 40)).isdisjoint(middle.train)  # labels reach into the test block
    assert 34 in middle.train
    # test labels run to row 64, so rows 60-64 are purged; the embargo then removes 65-67
    assert set(range(60, 68)).isdisjoint(middle.train) and 68 in middle.train


def test_walk_forward_windows():
    exp = walk_forward(100, train_size=40, test_size=20)
    assert [(s.train[0], s.train[-1], s.test[0], s.test[-1]) for s in exp] == [
        (0, 39, 40, 59),
        (0, 59, 60, 79),
        (0, 79, 80, 99),
    ]
    roll = walk_forward(100, train_size=40, test_size=20, expanding=False, embargo=5)
    assert all(len(s.train) == 40 and s.train[-1] + 5 < s.test[0] for s in roll)


# ---------------------------------------------------------------- selection


def test_selection_keeps_a_dominant_trial_and_finds_nothing_in_noise():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, (1500, 30))
    r[:, 4] += 0.003
    wf = select_out_of_sample(r, walk_forward(1500, 500, 250))
    assert (wf.chosen == 4).all()
    np.testing.assert_array_equal(wf.oos[500:], r[500:, 4])
    assert np.isnan(wf.oos[:500]).all()

    noise = rng.normal(0, 0.01, (2000, 50))
    best_in_sample = column_sharpe(noise).max()
    cp = cpcv_out_of_sample(noise, 6, 2)
    assert cp.oos.shape == (5, 2000) and not np.isnan(cp.oos).any()
    oos_sharpe = column_sharpe(cp.oos.T)
    assert best_in_sample > 0.04 and abs(oos_sharpe.mean()) < 0.02  # selection adds nothing


def test_overlapping_test_sets_are_rejected():
    r = np.random.default_rng(0).normal(0, 1, (60, 3))
    with pytest.raises(InvalidParameterError):
        select_out_of_sample(r, cpcv(60, 4, 2))


# ---------------------------------------------------------------- multiple testing


def test_haircut_reproduces_harvey_liu_example():
    # Harvey and Liu (2015): annual SR 0.75 over 20 years of monthly data (T = 240) gives
    # p = 0.0008 for one test; with 200 tests p_M = 0.15 and the haircut SR is 0.32 (~60%).
    h = haircut_sharpe(0.75, 20, 200, "independent")
    assert h.p_single == pytest.approx(0.0008, abs=5e-5)
    assert h.p_adjusted == pytest.approx(0.15, abs=0.005)
    assert h.sharpe == pytest.approx(0.32, abs=0.005)
    assert h.haircut == pytest.approx(0.57, abs=0.01)
    assert haircut_sharpe(0.75, 20, 200, "bonferroni").sharpe <= h.sharpe
    assert haircut_sharpe(0.75, 20, 1).sharpe == pytest.approx(0.75)


def test_holm_and_bhy_match_statsmodels():
    p = np.random.default_rng(5).uniform(0, 0.2, 40) ** 2
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1], rtol=1e-12)
    np.testing.assert_allclose(bhy(p), multipletests(p, method="fdr_by")[1], rtol=1e-12)
    cut = haircut_trials(np.array([0.2, 0.5, 1.0, 1.5]), years=10)
    assert (cut <= np.array([0.2, 0.5, 1.0, 1.5])).all() and cut[-1] > 0


def test_spa_and_stepm():
    noise = np.random.default_rng(101).normal(0, 0.01, (750, 20))
    assert data_snooping(noise, reps=300, seed=1).spa_pvalue > 0.05
    skill = np.random.default_rng(9).normal(0, 0.01, (750, 20))
    skill[:, 3] += 0.0015
    res = data_snooping(skill, reps=300, seed=0)
    assert res.spa_pvalue < 0.01 and 3 in res.better
