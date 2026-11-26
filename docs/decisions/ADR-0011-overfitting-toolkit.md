# ADR-0011: Tools for judging many backtests

- **Date:** 2026-11-26
- **Status:** accepted

## Context
The Pillar 4 study runs many strategy settings and must say which, if any, are better than
luck. The chosen scope (Pillar 4 research notes, 2026-11-26): Deflated Sharpe, PBO, SPA and
StepM, the Harvey-Liu haircut, purged k-fold and CPCV, plus a cost sweep. Every tool needs
conventions that are easy to get subtly wrong.

## Decisions

| Topic | Choice | Rejected alternative and why |
|---|---|---|
| Common input | A T x N matrix of daily trial returns from the backtester | Each tool re-running strategies: slower, and tools could disagree about the data |
| Sharpe units | Per period (per day) inside every formula; annualise only for display | Annualised inputs: the PSR and DSR formulas use the number of observations, so mixing units breaks them |
| Kurtosis | Non-excess (normal = 3), as in the PSR formula | Excess kurtosis: an off-by-3 error that silently changes every DSR |
| Effective number of trials | Participation ratio of the trial correlation eigenvalues, N^2 / sum(lambda^2) | Raw trial count: overstates independent bets for correlated grids. ONC clustering (Lopez de Prado): better for block-structured trials but adds a clustering step with its own choices; can be added later |
| PBO | CSCV on Sharpe ratios via per-block sums; S = 16 by default; leftover rows dropped from the start; PBO counts logits <= 0 (the paper's integral up to 0) | Recomputing each of the 12,870 splits from raw returns: correct but about 100x slower |
| CV splits | Contiguous groups; purge training rows whose label interval overlaps a test block's, then embargo the next rows | Shuffled k-fold: leaks the future through serial correlation |
| Embargo position | After the last purged row | Straight after the test block: overlaps the purge and then does nothing (found while testing) |
| Selection rule | Highest in-sample Sharpe ratio | Other scores can be added; Sharpe matches the DSR and PBO literature |
| Haircut Sharpe | Independent and Bonferroni from N; Holm and BHY from the p-values of every trial run | Harvey and Liu's correlated-test adjustment: relies on a simulation model fitted to published factors that cannot be reproduced exactly |
| SPA / StepM | `arch` package, stationary bootstrap, seeded; losses = negative returns | Own implementation: `arch` is maintained, documented, and needed for GARCH in Pillar 11 |
| Metrics | Sharpe and Sortino on excess returns over cash; Sortino downside deviation over all days; hit rate on invested days only | Sortino on raw returns: rewards holding cash |

## Consequences
- Every number in the study is reproducible from the trial matrix and a seed.
- The benchmark matters. A first run on 26 SPY moving-average rules against cash found almost
  all "significant", because long/flat SPY rules mostly earn the equity premium; the study
  must use buy-and-hold as the benchmark (decided in step 4).
- The participation ratio counted those 26 correlated rules as 1.7 effective trials, because
  one common factor (being long SPY) dominates. Whether to measure trials relative to the
  benchmark first is a step 4 decision.
