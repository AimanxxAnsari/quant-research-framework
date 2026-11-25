# ADR-0006: Validate every method on synthetic data first

- **Date:** 2026-11-25
- **Status:** accepted

## Context
Real data never reveals the true parameter, so an estimator or test that is biased, mis-sized or
leaking information can look plausible on real data. Several later pillars (pairs, backtesting,
risk, regimes) rely on estimators with known small-sample problems.

## Decision
- Every estimator and test gets a **known-answer** check: a generator with known parameters
  (GBM, OU, GARCH-t, cointegrated pairs, independent random walks, regime switching) and an
  assertion on recovery, size (false-positive rate) or calibration.
- A **study** (`qrf study pillar5`) quantifies the failure modes at realistic sample sizes and
  publishes them in the README, so later results are read against them.
- Generators take an explicit `numpy.random.Generator`; studies use fixed seeds.
- Look-ahead is tested by **planting** same-day information that a buggy implementation would
  exploit.

## Rejected
- Validating only on real data: cannot distinguish skill from estimator bias.
- Relying on library defaults without checking size: e.g. homoskedastic variance-ratio z rejects
  a true random walk about 25% of the time under GARCH.

## Consequences
- Each later pillar starts by adding generator-based tests for its methods.
- The summary table ("how often do methods fool you") is the reference for interpreting results.
