# ADR-0013: A local Streamlit app for the engine toggle and the study results

- **Date:** 2026-11-26
- **Status:** accepted

## Context
Pillar 4 built two backtest engines that must agree (ADR-0010). The plan asked for a page that
switches between them, or runs both and shows each engine's analysis side by side with a
comparison. Prices come from Tiingo and FRED and may not be redistributed (ADR-0002), and the
Pillar 4 study is pre-registered, so anything interactive must not blur what the study found.

## Decisions

| Topic | Choice | Rejected alternative and why |
|---|---|---|
| Where it runs | Locally, `qrf app`; a seeded synthetic market works from a clean clone with no keys, and "my data" uses the local warehouse | Streamlit Community Cloud: could only show the demo and the committed statistics, and adds secrets and a public URL to maintain |
| Pages | Engine lab (one rule on one asset, engine set to weights, loop or both) and a read-only study explorer over `reports/pillar4` | A validation page that runs DSR and PBO on grids chosen in the app: twice the work and an easy way to go fishing |
| Charts | Plotly, coloured from `plotting.py`, light or dark from the Streamlit theme | Streamlit's own charts: no shared palette. Matplotlib images: no hover or zoom |
| Code layout | Logic in `app/lab.py`, `app/study.py`, `app/charts.py` with no Streamlit import; `app/main.py` only lays out widgets | Logic inside the Streamlit script: only testable through the UI |
| Dependencies | Optional `app` dependency group (Streamlit, Plotly) | Main dependencies: every user of the library would install a web server |
| Tests | Unit tests on the logic, plus Streamlit's `AppTest` running the page headless in CI (toggle to "both", currency settings, study page) | Browser tests: heavier, and not needed to prove the page runs |
| Honesty | The lab says on every run that it is in-sample exploration; the study page only reads committed files | Letting the lab write to `reports/`: would mix exploration with the pre-registered results |

## Consequences
- "Both" runs the rule and its benchmark on each engine and shows two full analyses, then a
  comparison: metrics with their difference, the NAV gap over time, run times, and the
  look-ahead check.
- The loop engine asks the rule for a decision on each bar's history, so its run time grows with
  the square of the history length for rules that walk their whole history (about 30 s for the
  filter rule on NZD since 1985, 3 s from 2015). The "my data" mode has a first-year setting;
  the reference engine was not changed to make the page faster.
- Results are cached per input, so switching back to a setting already run is instant.
