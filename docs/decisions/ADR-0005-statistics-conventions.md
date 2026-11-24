# ADR-0005: Statistics conventions

- **Date:** 2026-11-24
- **Status:** accepted

## Context
Small convention choices (return type, sampling day, price adjustment, test corrections) change
results. They must be fixed once and documented.

## Decisions
| Topic | Choice | Rejected alternative and why |
|---|---|---|
| Return type for statistics | Log returns | Simple returns: do not add across time; kept for portfolio aggregation |
| Prices for returns and ranges | Split/dividend-adjusted OHLC | Raw prices: dividends and splits create false jumps and false overnight gaps |
| Weekly sampling | Wednesday close (`W-WED`) | Friday: more holidays; month-end: too few observations |
| Annualisation | 252 trading days; sqrt-time for volatility | Calendar days: mixes non-trading days; sqrt-time is noted as an iid assumption |
| Volatility window | 21 trading days (one month) | Longer windows: slower to react; configurable |
| EWMA | lambda 0.94, value at t uses data to t-1 | Including day t: look-ahead for any forecasting use |
| Variance-ratio inference | Heteroskedasticity-robust z* | Homoskedastic z: over-rejects under volatility clustering |
| Many tests | Holm correction across all tests in a study, count reported | Uncorrected p-values: guaranteed false discoveries at 44 tests |
| Hurst exponent | R/S read against a shuffled baseline + Lo's modified R/S | Raw R/S vs 0.5: biased upward and fooled by short memory |
| FX | Always USD per unit of foreign currency (inverting where needed) | Native FRED quotes: mixes conventions and flips signs |
| Non-synchronous data | Report daily, next-day, lag-corrected and weekly correlation | Daily correlation alone: biased towards zero when closes differ |

## Consequences
Every study states these conventions implicitly by using the shared functions; changes need a new
ADR.
