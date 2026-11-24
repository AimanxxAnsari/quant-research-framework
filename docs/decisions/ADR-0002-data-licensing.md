# ADR-0002: Data licensing and what is committed

- **Date:** 2026-11-24
- **Status:** accepted

## Context
The repository is public. Sources have different terms: Treasury data is public; FRED series may
carry individual copyright notices; the French library is free for research; Tiingo's free tier is
for internal use only.

## Decision
- **Never commit data.** `data/`, `*.parquet` and `*.duckdb` are git-ignored. The repo contains
  code, configuration, tests on *synthetic* payloads that mimic each format, and documentation.
- **Publish only derived results** (statistics, charts) in write-ups.
- Every source is listed in `config/sources.yaml` with access route, terms URL and redistribution
  rule.
- **Secrets** live in a git-ignored `.env`; keys are sent in headers where the API allows (Tiingo)
  and are redacted from URLs in logs and the raw-store manifest (FRED requires the key in the query).

## Consequences
- Others reproduce results by running `qrf refresh` with their own free keys.
- Tests never need network or keys.
