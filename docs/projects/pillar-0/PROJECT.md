# Pillar 0: a reproducible, point-in-time data foundation

## Abstract
Every later result in this framework depends on data that is complete, correctly timestamped and
traceable. Pillar 0 builds a small but production-shaped data layer: collectors for FRED,
US Treasury, the Kenneth French library and Tiingo; an immutable, content-hashed raw archive;
deterministic clean tables in Parquet with a DuckDB warehouse; and data-quality checks that fail
loudly. The first build produced about 408,000 observations across 45 series (1947 to date) and
197,000 daily price rows for 33 ETFs (2000 to date), with checks catching one real data event
and one access failure.

## 1. Why this matters
Most backtest errors are data errors: look-ahead from using data before it was published,
survivorship, silent gaps, stale series, or a vendor revising history. Data engineering in
trading firms exists to prevent these. This pillar applies the same principles at personal scale.

## 2. Design principles
1. **Raw is immutable.** Downloads are stored as received, hashed, never edited (ADR-0003).
2. **Clean is derived.** Clean tables are a pure function of the raw layer, so rebuilding is
   deterministic.
3. **Lineage on every row.** `source_id`, `retrieved_at`, `raw_sha256`.
4. **Fail visibly.** Coverage, uniqueness, missing values, ranges, OHLC consistency, jumps and
   freshness are checked on every refresh (ADR-0004).
5. **Licensing respected.** No vendor data in the repository (ADR-0002).

## 3. Data
See the README table. FX series keep their native quote convention (USD per EUR vs JPY per USD),
recorded in the instrument master.

## 4. Results of the first build (2026-11-24)
| Table | Rows | Span |
|---|---|---|
| observations (FRED) | 280,490 | 1947-01 to 2026-11-23 (series-dependent) |
| observations (Treasury) | 127,883 | 1990-01-02 to 2026-11-23 |
| prices (Tiingo) | 196,971 | 2000-01-03 to 2026-11-23 |

**What the checks caught**
- **OVX = 226 on 2020-04-20** failed the volatility range check. This was a real market event
  (the day WTI futures settled negative; OVX later exceeded 300), not an error, so the bound was
  widened with a comment. It shows why bounds must flag impossible values, not unusual ones.
- **French library: HTTP 403.** The collector failed visibly, coverage check FAIL, other sources
  unaffected. Open item.

## 5. Limitations
- No ALFRED vintages yet (macro revisions); no release-time stamps.
- One price vendor; no cross-vendor reconciliation.
- Daily granularity only.

## 6. Next
Pillar 2 statistics toolkit on top of these tables.
