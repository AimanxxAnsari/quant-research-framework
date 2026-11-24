# ADR-0001: Tooling and layout

- **Date:** 2026-11-24
- **Status:** accepted

## Context
One repository holds Pillars 0 and 2-13, which share a data layer, statistics and a backtester.
It must be reproducible, testable offline in CI, and light enough for one person.

## Options considered
| Topic | Options | Choice |
|---|---|---|
| Environment | pip / conda / Poetry / **uv** | uv + `uv.lock`, Python 3.12 (same as trade-analytics) |
| Storage | CSV / SQLite / Postgres / **Parquet + DuckDB** | Columnar, fast, file-based, SQL without a server |
| Transforms | **plain Python + DuckDB SQL** / dbt-duckdb | Start plain; add dbt only when SQL models multiply |
| Orchestration | Airflow / Prefect / **make + CLI** | `make refresh` -> `qrf refresh`; Airflow later if scheduling is needed |
| Layout | flat / **src/** | `src/quant_research`, tests run against the installed package |

## Decision
As in the table. Package `quant_research`, console command `qrf`, Makefile targets
`install test lint fetch build check refresh`.

## Consequences
- A full refresh is one command; CI runs lint + offline tests only (no keys in CI).
- Moving to dbt or Airflow later is a new ADR, not a rewrite: collectors and checks are plain
  functions.
