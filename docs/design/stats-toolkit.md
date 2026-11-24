# Design: statistics toolkit (Pillar 2)

**Status:** accepted · 2026-11-24 · Packages: `quant_research.stats`, `quant_research.data.access`,
`quant_research.monitor`, `quant_research.studies.pillar2`

## Purpose
Tested building blocks that every later pillar reuses, plus a first set of research results.

## Modules
| Module | Functions |
|---|---|
| `data.access` | `series`, `prices`, `ohlc`, `fx_usd_per_foreign` - wide, date-indexed frames, no forward fill |
| `stats.returns` | simple/log returns, cumulative, CAGR, annualised vol, weekly resampling, drawdown, max drawdown with peak/trough/recovery |
| `stats.volatility` | close-to-close, EWMA, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang; `all_estimators` |
| `stats.distribution` | moments, Jarque-Bera, tail frequency vs normal |
| `stats.dependence` | beta with SE and R^2, rolling beta/correlation, lead-lag correlation, lagged-covariance correction |
| `stats.random_walk` | Lo-MacKinlay variance ratio (z, z*), R/S Hurst, shuffled baseline, Lo modified R/S |
| `monitor` | cross-asset snapshot: levels, 1W/1M/3M/YTD changes, realised vol and percentile |
| `reporting` | README marker injection, Markdown tables |
| `studies.pillar2` | the study behind the README results; all commentary computed from the numbers |

## Invariants
- Functions are pure; data access is separate from statistics.
- No forward-filling inside access: callers align explicitly (`dropna` on the joined frame).
- EWMA and any forecast-type output at date t use information up to t-1 only.
- README results are generated, never edited by hand.

## Test plan (known answers)
| Function | Known answer |
|---|---|
| returns / drawdown | hand-computed paths |
| EWMA | hand-computed recursion |
| range estimators | simulated intraday GBM with known sigma: all recover ~20%; GK less noisy than Parkinson, Parkinson less than close-to-close |
| Yang-Zhang | simulated overnight gaps: recovers total variance; Garman-Klass misses the gap |
| describe | normal vs Student-t(3) samples |
| beta | y = 2x + noise |
| lead-lag | y reacting one day later peaks at lag 1 |
| non-synchronous | half-day factor model: daily corr ~0.5, weekly > 0.85, lagged-covariance correction ~1 |
| variance ratio | iid ~1; AR(1) phi=0.3 -> 1.3; MA(1) -> 0.5; z* large for AR(1) |
| Hurst | iid near shuffled baseline; AR(1) phi=0.7 fools R/S but not Lo's test |
