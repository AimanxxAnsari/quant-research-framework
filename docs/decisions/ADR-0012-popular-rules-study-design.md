# ADR-0012: Design of the popular-rules study

- **Date:** 2026-11-26
- **Status:** accepted (pre-registered: tag `pillar4-preregistration`)

## Context
Pillar 4 ends with a study of whether widely shared technical trading rules beat a passive
benchmark once timing, costs and the number of rules tried are taken into account. The
literature (Sullivan, Timmermann and White 1999; Park and Irwin 2007; Bajgrowicz and Scaillet
2012) says few survive; the study tests that with this repo's own engines and toolkit, and
tests the toolkit itself.

## Decisions

| Topic | Choice | Rejected alternative and why |
|---|---|---|
| Rules | 14 families: 7 daily-feasible je-suis-tm indicators, a daily Dual Thrust, and classic academic rules (MA crossover, price vs MA, channel breakout, filter rule, momentum, stochastic) | je-suis-tm only: leaves out the rules the literature tested. A large academic universe: heavier and harder to read |
| Simplified rules | RSI and Bollinger as band rules; shooting star as "step aside after the pattern" for long/flat ETFs | The pattern-recognition versions: subjective and hard to code without hindsight |
| Dropped | London Breakout and intraday Dual Thrust | Need intraday data we do not have |
| Universe | 33 ETFs and 9 currencies | ETFs only: FX trend rules are a well-known case and the engine already supports carry |
| Direction | ETF long/flat, FX long/short | Shorting ETFs: borrow costs not modelled |
| Benchmark | ETF: buy-and-hold of the same ETF under the same fills and costs. FX: cash | Cash for ETFs: a smoke test showed nearly every SPY timing rule "beats" cash by earning the equity premium |
| Statistic | Annualised Sharpe of daily active returns | Raw Sharpe: rewards exposure to the asset rather than timing |
| Honesty | Pre-registration committed and tagged before any result; clarifications and additions logged with dates; development to 2021-11-30, holdout sealed until one final run | No pre-registration: no evidence the tests were fixed in advance |
| Holdout guard | The development run loads markets only up to 2021-11-30, so holdout prices are never read | A flag checked at evaluation time: easier to bypass by mistake |
| Reproducibility | Every run writes a manifest: commit, uncommitted files, data SHA-256, data-through date, package versions; daily returns stay in `data/` (git-ignored); only per-trial statistics are committed | Committing daily return series: derived from vendor prices, which may not be redistributed |
| Engine check | A seeded 5% sample re-run on the loop engine | All trials on the loop engine: hours instead of minutes |

## Consequences
- The headline is a null result if the tests say so, published as plainly as a positive one.
- Positive active returns on ETFs can come from avoiding an asset that lost money; the report
  shows buy-and-hold return next to the share of rules beating it (added after seeing results,
  logged as a deviation).
- The holdout run is a separate, one-time step after the development results are committed.
