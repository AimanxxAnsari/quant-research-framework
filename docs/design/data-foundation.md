# Design: data foundation (Pillar 0)

**Status:** accepted · 2026-11-24 · Package: `quant_research.data`

## Purpose
One reproducible, point-in-time-aware data layer that every later pillar reads from.

## Components
| Module | Responsibility |
|---|---|
| `settings` | repo root, data paths (`QRF_DATA_DIR`), config loading, secrets from `.env` |
| `http` | GET with timeout, retries + exponential backoff on 429/5xx/network errors, no retry on other 4xx, URL redaction |
| `raw_store` | immutable snapshots + manifest (ADR-0003) |
| `collectors/*` | `fetch()` (network -> raw store) and `parse()` (bytes -> tidy frame, pure) per source |
| `build` | latest snapshots -> clean Parquet tables + DuckDB views, atomic writes |
| `quality` | checks -> PASS/WARN/FAIL table and `health.md` (ADR-0004) |
| `cli` | `qrf fetch / build / check / refresh` |

## Tables
**observations** (long): `series_id, date, value, source_id, retrieved_at, raw_sha256`.
One table for all scalar time series keeps joins and checks uniform.

**prices**: `ticker, date, open, high, low, close, volume, adj_open, adj_high, adj_low,
adj_close, adj_volume, div_cash, split_factor` + lineage. Raw prices for anything tied to traded
levels; adjusted prices for returns.

**series**: catalogue with name, asset class, unit, frequency (from `config/instruments.yaml`).

## Source specifics
| Source | Unit of download | Notes |
|---|---|---|
| FRED | one JSON per series, full history | "." means missing (holidays) and is dropped; FX quote conventions differ by series (USD per EUR vs JPY per USD) and are recorded in the instrument master |
| Treasury | one CSV per curve per year | tenor headers vary ("1 Mo", "1.5 Month"); normalised to `UST_PAR_1M`, `UST_PAR_1.5M`... |
| French | one ZIP per dataset | text preamble and footer around the daily block; values in percent |
| Tiingo | one JSON per ticker from 2000 | token in the Authorization header; free tier ~50 requests/hour, so a full refresh (33 tickers) fits once per hour |

## Invariants
- Clean tables are a pure function of the raw layer (idempotent build, tested).
- Every clean row carries lineage to its raw snapshot.
- No secrets in code, manifests, logs or the repository.
- Failed sources are reported per source; other sources still refresh.

## Known gaps / next steps
- French library download returns HTTP 403 to this client (see devlog 2026-11-24).
- Point-in-time macro (ALFRED vintages) and release calendars: Pillar 8 territory.
- Exchange calendars per instrument; ECB / BoE / CFTC / EIA collectors when pillars need them.

## Test plan
Offline tests with synthetic payloads in each source's format: parsers; raw store (read-only,
dedup, new snapshot on change, tamper detection, URL redaction); build idempotency and lineage;
planted errors (99% yield -> range FAIL; low > close -> OHLC FAIL; +150% day -> jump WARN;
missing ticker -> coverage FAIL; stale data -> freshness WARN).
