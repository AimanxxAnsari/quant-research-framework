# Design index

Design docs describe how each component works. Decision records (in
[`../decisions/`](../decisions/)) record why, including the options that were rejected.

## Design docs

| Component | Doc | Pillar |
|---|---|---|
| Collectors, raw store, clean tables, quality checks | [data-foundation.md](data-foundation.md) | 0 |
| Returns, volatility, distributions, dependence, random-walk tests | [stats-toolkit.md](stats-toolkit.md) | 2 |
| Synthetic generators and known-answer tests | [validation.md](validation.md) | 5 |

## Key decisions and trade-offs

| ADR | Decision | Main trade-off or rejected alternative |
|---|---|---|
| [0001](../decisions/ADR-0001-tooling-and-layout.md) | uv, src layout, DuckDB + Parquet, plain Python | dbt and Airflow deferred: more moving parts than one person needs now |
| [0002](../decisions/ADR-0002-data-licensing.md) | Never commit data; publish derived results only | Results reproduce with the reader's own free keys rather than from frozen data |
| [0003](../decisions/ADR-0003-immutable-raw-store.md) | Immutable, hashed raw snapshots with a manifest | More disk than overwriting, in exchange for lineage and revision history |
| [0004](../decisions/ADR-0004-quality-checks.md) | Own PASS / WARN / FAIL checks, non-zero exit on FAIL | Pandera and dbt tests rejected for now; ranges are sanity bounds, not opinions |
| [0005](../decisions/ADR-0005-statistics-conventions.md) | Log returns, adjusted prices, 252 days, robust tests, Holm correction | Simple returns and homoskedastic tests rejected for statistics |
| [0006](../decisions/ADR-0006-validate-on-synthetic-data-first.md) | Every method is checked on synthetic data with a known answer | Validating on real data alone cannot tell skill from estimator bias |
| [0007](../decisions/ADR-0007-engineering-standards.md) | Error family, strict typing on core maths, pre-commit, 3.12/3.13 CI | Strict typing everywhere rejected: pandas stubs make study scripts noisy |
| [0008](../decisions/ADR-0008-short-rates-for-fx-carry.md) | Daily overnight rates for USD, EUR, GBP; monthly OECD rates for the other seven currencies | ECB policy-rate proxy for EUR rejected once EONIA was found on FRED; monthly carry is approximate |
