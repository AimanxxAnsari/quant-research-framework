# quant-research-framework

**A cross-asset quant research framework built on a point-in-time, reproducible data foundation.**

An independent quant research project covering rates, FX, equity, commodity and volatility data:
statistics, energy-futures term structure, a backtesting engine with overfitting controls,
synthetic-data validation, relative value and statistical arbitrage, systematic strategies, risk
(VaR / ES / EVT / copulas), portfolio construction and yield-curve models, volatility forecasting
and regime models. Every number is meant to be traceable to raw data, code and assumptions.

## Roadmap

| Pillar | Topic | Progress |
|---|---|---|
| 0 | Data foundation: collectors, immutable raw store, clean tables, quality checks | Built |
| 2 | Returns, volatility, correlation, variance ratio, Hurst | Next |
| 3 | Energy futures: contract mapping, roll engine, curves, convenience yield, Schwartz-Smith | Planned |
| 4 | Backtesting engine, costs, walk-forward, purged CV, Deflated Sharpe, PBO | Planned |
| 5 | Synthetic data and known-answer tests | Planned |
| 6 | Pairs, cointegration, Kalman hedge ratios, PCA statistical arbitrage | Planned |
| 7 | Trend, carry, cross-sectional momentum, FX forward-premium regression | Planned |
| 9 | Risk engine: VaR / ES, backtests, stress tests, EVT, copulas | Planned |
| 10 | Portfolio construction, factor models, yield-curve PCA, Nelson-Siegel-Svensson, short-rate models | Planned |
| 11 | Volatility clustering, GARCH family, HAR-RV, variance risk premium | Planned |
| 12 | Forecast ledger and calibration | Planned |
| 13 | Regime models: HMM, Markov switching, change points | Planned |

## Pillar 0: data foundation

```
collectors (FRED, Treasury, French, Tiingo)
        |  HTTP with retries; secrets in .env, redacted from every log and manifest
        v
data/raw/   immutable snapshots: <source>/<dataset>/<time>_<sha12>.<ext> + manifest.jsonl
        |  latest snapshot per dataset, parsed by pure functions
        v
data/clean/ observations.parquet | prices.parquet | series.parquet  + warehouse.duckdb (views)
        |
        v
quality checks: coverage, uniqueness, missing values, plausible ranges, OHLC consistency,
                jumps, freshness  ->  data/health.md, non-zero exit on any FAIL
```

| Source | Content | Coverage at first build |
|---|---|---|
| FRED | 26 series: Treasury constant-maturity yields, fed funds, SOFR, real yield, breakeven, VIX / OVX / GVZ, 9 H.10 FX rates, WTI and Brent spot, CPI, unemployment | from 1947 (series-dependent) to the latest release |
| US Treasury | Daily par nominal (from 1990) and real (from 2003) yield curves, 19 tenors | 1990 to latest |
| Kenneth French library | Daily Fama-French 3 factors and momentum | collector built; download currently blocked (HTTP 403), see devlog |
| Tiingo | 33 ETFs across US and international equity, sectors, bonds, commodities, FX; raw and adjusted OHLCV | 2000 to latest |

Design: [`docs/design/data-foundation.md`](docs/design/data-foundation.md).
Decisions: [`docs/decisions/`](docs/decisions/).

**Data is never committed.** Raw vendor data and derived tables stay local
([ADR-0002](docs/decisions/ADR-0002-data-licensing.md)); the repository holds code, configuration,
tests on synthetic payloads, and documentation.

## Run

```bash
cp .env.example .env          # add FRED_API_KEY and TIINGO_API_KEY (free)
make install
make refresh                  # fetch -> build -> check  (or: uv run qrf refresh)
make test
```

Query with SQL:

```python
import duckdb

con = duckdb.connect("data/warehouse.duckdb", read_only=True)
con.sql("SELECT date, value FROM observations WHERE series_id = 'DGS10' ORDER BY date DESC LIMIT 5")
```

## Documentation

- [Pillar 0 write-up](docs/projects/pillar-0/PROJECT.md)
- [Design docs](docs/design/)
- [Decision records](docs/decisions/)
- [Development log](docs/devlog/)
- [Learning notes](docs/learning/)
- [References](references.bib)
