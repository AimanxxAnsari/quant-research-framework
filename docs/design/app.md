# Engine lab and study explorer (Streamlit)

Run: `uv sync --group app`, then `uv run qrf app`. Decisions: [ADR-0013](../decisions/ADR-0013-engine-lab-app.md).

## Modules

| Module | What it does | Streamlit? |
|---|---|---|
| `app/lab.py` | The 14 rule families and their parameters, a seeded demo market, single-asset slicing, `run_lab` (rule and benchmark on each chosen engine, timed), `LabRun.comparison` and `nav_gap` | No |
| `app/study.py` | Loads `reports/pillar4/development` and `holdout`; filters trials | No |
| `app/charts.py` | Plotly figures in the repo palette, light or dark | No |
| `app/main.py` | Sidebar inputs, the two pages, caching | Yes |

## Engine lab
1. Inputs: market (demo or my data), asset, rule family and parameters, shorts (currencies
   only, as in the study), fill rule, cost multiplier, first year (my data only), engine.
2. `run_lab` runs the rule with the configured look-ahead check, and the benchmark
   (buy-and-hold for an ETF, cash for a currency) on the same engine with the check off.
   Metrics come from `backtest.summary`, plus the active Sharpe ratio against the benchmark.
3. One engine: metrics table, growth of 1, drawdown and position. Both: those panels for each
   engine in two columns, then the comparison (metrics, their difference, NAV gap over time, run
   time, look-ahead check).

## Demo market
Seeded, about six years of business days: `TREND` (an ETF drifting up), `DECAY` (an ETF losing
about a quarter a year, like a futures ETF with heavy roll costs), and `FXA` / `FXB`
(currencies with moving short rates). `DECAY` reproduces the study's main lesson: timing rules
"beat" it by being out of it, not by timing.

## Study explorer
Headline tests (trial count, SPA p-values, survivors), the distribution of development active
Sharpe ratios, a filterable trial table, the per-asset table, and the holdout results. It reads
committed files only, so it shows exactly what the README shows.

## Tests
`tests/test_app.py`: every pre-registered rule rebuilds from its family and parameters; engines
agree on the demo market; inputs are validated; charts build in both themes; and `AppTest`
runs the page headless (default run, the "both" toggle, a currency, the study page).
