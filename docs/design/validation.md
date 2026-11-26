# Design: synthetic data and known-answer validation (Pillar 5)

**Status:** accepted · 2026-11-25

## Modules
| Module | Contents |
|---|---|
| `validation.synthetic` | `gbm_log_prices`, `ou` (exact AR(1) sampling from half-life), `garch11` (Student-t, returns + true vol), `cointegrated_pair`, `independent_random_walks`, `regime_switching`, `stationary_distribution` |
| `stats.mean_reversion` | `ar1_fit` (phi, half-life), `ols_levels` (naive levels regression), `engle_granger` (statsmodels `coint`) |
| `risk.var` | `historical_var`, `ewma_normal_var` (forecasts use data to t-1), `breaches`, `kupiec_pof` |
| `validation.overfitting` | `expected_max_sharpe` (Bailey & Lopez de Prado), `annualised_sharpe`, `ma_crossover_returns` (signal at t, position over t+1) |
| `studies.pillar5` | half-life bias, cointegration size/power/screening, variance-ratio size under GARCH, VaR calibration, max-Sharpe vs trials and an MA grid in/out of sample |

## Known-answer tests
| Check | Expected |
|---|---|
| GBM log-return mean and vol | (mu - sigma^2/2), sigma |
| OU half-life, 100k steps | 10 within 5% |
| OU half-life, 250 steps | median below 10 (Kendall bias) |
| GARCH unconditional variance | omega / (1 - alpha - beta); squared returns autocorrelated |
| Cointegrated pair | Engle-Granger rejects, beta ~1.5 |
| Independent random walks | naive levels t "significant" |
| Regime switching | state shares = stationary distribution; stay probability 0.99 |
| Kupiec | LR = 0 at exactly the target rate; rejects at 2.5x |
| VaR with true normal vol | 1% breaches |
| Expected max Sharpe | matches simulated max of N normals |
| MA crossover look-ahead | planted same-day signal earns ~0 Sharpe (a look-ahead bug would earn > 10) |

## Runtime
Full study about 15 seconds; a smoke test with tiny repetition counts runs in CI.

---

# Pillar 4 additions: tools for judging many backtests (2026-11-26, ADR-0011)

## Modules
| Module | Contents |
|---|---|
| `validation.sharpe` | `sharpe_stats`, `probabilistic_sharpe`, `deflated_sharpe`, `min_track_record_length`, `effective_number_of_trials` (all Sharpe ratios per period) |
| `validation.pbo` | `cscv`: PBO, logits, performance degradation slope, probability of OOS loss |
| `validation.splits` | `walk_forward` (expanding or rolling, embargo), `purged_kfold`, `cpcv`, `cpcv_paths`, `purged_train` |
| `validation.selection` | `select_out_of_sample` (walk-forward / k-fold: pick the IS winner, stitch its OOS returns), `cpcv_out_of_sample` (OOS paths) |
| `validation.multiple_testing` | `haircut_sharpe` (independent, Bonferroni), `holm`, `bhy`, `haircut_trials`, `data_snooping` (SPA and StepM from `arch`) |
| `backtest.summary` | full metric set and `cost_sweep` (0x, 1x, 2x, 3x costs) |

All of them take a T x N matrix of daily trial returns, so a study runs each strategy setting
through the backtester once and every tool reads the same matrix.

## Known-answer tests
| Check | Expected (source) |
|---|---|
| DSR, N = 100, annual SR 2.5, T = 1250, skew -3, kurtosis 10, V[SR] = 1/2 | SR0 0.1132, DSR 0.9004 (Bailey and Lopez de Prado 2014) |
| Same, N = 46 | DSR 0.9505 (same paper) |
| Same with normal returns | DSR >= 0.95 up to N = 88, below at 89 (same paper) |
| PSR with normal returns | Phi(SR sqrt(T-1) / sqrt(1 + SR^2/2)) (Lo 2002) |
| Minimum track record length | PSR crosses the target between floor and ceiling of MinTRL |
| Effective trials | 20 independent -> 20; 12 trials in 3 identical clusters -> 3; identical -> 1 |
| CSCV split count, S = 16 | 12,870 (the paper prints 12,780, a typo) |
| PBO with one clearly skilled trial | 0 |
| PBO on pure noise | 0.5 on average over 200 datasets (one dataset ranges 0.3 to 0.9) |
| CPCV, 6 groups, 2 test groups | 15 splits, 5 complete paths |
| Purging with 5-day labels and embargo 3 | the 5 rows before and the 5 after a test block are purged, the next 3 embargoed |
| Haircut, annual SR 0.75, 240 months, 200 tests | p 0.0008 -> 0.15, haircut SR 0.32 (Harvey and Liu 2015) |
| Holm and BHY | equal to statsmodels `holm` and `fdr_by` |
| SPA / StepM | noise: p > 0.05; one skilled trial: p < 0.01 and found by StepM |
| Cost sweep | cost drag scales 2x and 3x; Sharpe falls with costs |

## Runtime
CSCV with S = 16 (12,870 splits) on 26 trials x 6,763 days: under a second, using per-block
sums. The full set of tools on that matrix: a few seconds.
