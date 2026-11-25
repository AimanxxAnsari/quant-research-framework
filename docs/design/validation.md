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
