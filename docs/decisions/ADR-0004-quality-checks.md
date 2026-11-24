# ADR-0004: Data-quality checks

- **Date:** 2026-11-24
- **Status:** accepted

## Context
Stale or broken data must fail visibly instead of silently feeding research.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| pandera schemas | declarative | another dependency; less natural for cross-table and freshness checks |
| dbt tests | standard in industry | requires adopting dbt now (deferred, ADR-0001) |
| **own check functions** | simple, explicit, easy to unit-test with planted errors | must maintain |

## Decision
Own checks in `quality.py`, each returning PASS / WARN / FAIL with a detail message:
coverage, key uniqueness, missing values, plausible ranges by asset class, OHLC consistency
(FAIL); large adjusted jumps and staleness (WARN). `qrf check` writes `data/health.md` and exits
non-zero on any FAIL.

**Ranges are sanity bounds, not opinions:** they flag impossible values (a 99% Treasury yield),
not unusual ones. When a real extreme trips a bound, the bound is widened with a comment
(see devlog: OVX at 226-325 in April 2020).

## Consequences
- A deliberately broken input makes a test fail (planted 99% yield; inconsistent OHLC bar).
- Migrating these checks to dbt tests later is straightforward.
