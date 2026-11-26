# Pre-registration: popular trading rules after overfitting correction

**Status:** approved 2026-11-26 and tagged
`pillar4-preregistration`, before any study result exists. Any later change is a new, dated
section at the end; nothing above it is edited.

## 1. Question
Do widely shared technical trading rules beat a passive benchmark on liquid ETFs and major
currencies once we allow for (a) realistic timing and costs, (b) the number of rules and
settings tried, and (c) a period that was never used to design or select them?

**Expected answer, stated in advance:** few or none survive. This is what the literature finds
(Sullivan, Timmermann and White 1999; Bajgrowicz and Scaillet 2012; Park and Irwin 2007), and
the study is a test of the toolkit as much as of the rules. A null result will be reported as
plainly as a positive one.

## 2. Data
- **ETFs (33):** SPY, QQQ, IWM, RSP, EFA, VGK, EWJ, EEM, EWA, EWC, XLE, XLF, XLK, XLV, XLU, XLP,
  XLI, TLT, IEF, SHY, TIP, LQD, HYG, GLD, SLV, GDX, USO, UNG, DBC, DBA, CPER, UUP, FXE. Tiingo
  split- and dividend-adjusted daily open, high, low and close.
- **Currencies (9):** EUR, GBP, AUD, NZD, JPY, CHF, CAD, NOK, SEK against USD. FRED H.10 noon
  fixing, expressed as USD per unit of foreign currency.
- **Rates:** short rates per ADR-0008 (cash return and FX carry).
- **Period:** each asset from its first date with prices and rates. **Development period ends
  2021-11-30. Holdout: 2021-12-01 to the last date in the warehouse when the holdout is opened.**
- Live data: results state their "data through" date and reproduce with the reader's own free
  keys.

## 3. Rules
Each trial is one rule, one parameter setting and one asset. The rule outputs a target
position at each close using data up to that close only; the position is +1 (long 100%), 0
(flat) or, for currencies only, -1 (short 100%). ETFs are long/flat; currencies long/short.
For long/flat ETFs a rule's "short" signal means flat.

Rules marked **OHLC** need high, low or open prices and run on ETFs only (FX has one fixing a
day). The others use closes and run on both.

| # | Family | Source | Definition (signal at close t) | Grid | Trials per asset |
|---|---|---|---|---|---|
| 1 | Moving-average crossover | Brock et al. 1992 | long if SMA(fast) > SMA(slow), else short/flat | fast {5, 10, 20, 50} x slow {50, 100, 200}, fast < slow | 11 |
| 2 | Price vs moving average | Brock et al. 1992 | long if close > SMA(n), else short/flat | n {50, 100, 200} | 3 |
| 3 | Channel breakout | Brock et al. 1992 | long when close > highest close of the previous n days; short/flat when close < lowest close of the previous n days; otherwise keep the last position | n {20, 50, 100, 200} | 4 |
| 4 | Filter rule | Alexander 1961 | long after a rise of x% from the lowest close since the last signal; short/flat after a fall of x% from the highest close since the last signal | x {1, 2, 5, 10}% | 4 |
| 5 | Time-series momentum | Moskowitz et al. 2012 | long if the return over the past L days > 0, else short/flat | L {21, 63, 126, 252} | 4 |
| 6 | MACD | je-suis-tm | long if EMA(fast) - EMA(slow) > its EMA(signal), else short/flat | (12, 26, 9), (8, 17, 9), (5, 35, 5) | 3 |
| 7 | RSI | je-suis-tm (simplified) | Wilder RSI(p): long when RSI < L until RSI > 50; short (FX) when RSI > U until RSI < 50; else flat | p {7, 14} x (L, U) {(30, 70), (20, 80)} | 4 |
| 8 | Bollinger bands | je-suis-tm (simplified) | long when close < SMA(n) - k sd until close > SMA(n); short (FX) when close > SMA(n) + k sd until close < SMA(n); else flat | n {20, 50} x k {1.5, 2, 2.5} | 6 |
| 9 | Parabolic SAR | je-suis-tm, Wilder 1978 | long while close is above the SAR, else flat. **OHLC** | (step, max) (0.02, 0.2), (0.01, 0.1), (0.04, 0.4) | 3 |
| 10 | Heikin-Ashi | je-suis-tm | long after k consecutive rising Heikin-Ashi candles; flat after k consecutive falling ones; else keep. **OHLC** | k {1, 2, 3} | 3 |
| 11 | Awesome Oscillator | je-suis-tm | long if SMA(fast) - SMA(slow) of the median price (high + low) / 2 > 0, else flat. **OHLC** | (5, 34), (10, 50) | 2 |
| 12 | Shooting star | je-suis-tm | long by default; flat for h days after a shooting star (upper shadow >= r x body, lower shadow <= body, after a close above SMA(20)). **OHLC** | r {2, 3} x h {5, 10, 20} | 6 |
| 13 | Dual Thrust (daily) | je-suis-tm, adapted | range = max(highest high - lowest close, highest close - lowest low) over the previous n days; long if close > open + k x range; flat if close < open - k x range; else keep. **OHLC** | n {5, 10} x k {0.3, 0.5, 0.7} | 6 |
| 14 | Stochastic oscillator | common | long if %K(n) > %D (3-day SMA of %K), else flat. **OHLC** | n {14, 21} | 2 |

Not included: London Breakout (needs the London session open; intraday data) and the
original intraday Dual Thrust. Trials: 61 per ETF x 33 = 2,013; 39 per currency x 9 = 351;
**2,364 in total**. Every trial is run and reported, whatever its result.

Each rule is written for this repo (no code copied, ADR on the sources); look-ahead is
checked on every rule by the engine's truncation check.

## 4. Execution
- Weights engine, `next_open` fills as the headline (FX fills at the next fixing); `next_close`
  reported alongside. Costs: ETF 2 bp and FX 1 bp one way (ADR-0009), swept at 0x, 1x, 2x, 3x.
- Cash and FX balances earn their short rates. Initial position all cash.
- Engine check: a random 5% of trials (seed 2026) are re-run on the loop engine; the largest
  NAV gap is reported.

## 5. What is measured
- **Active return** of a trial, per day: ETF trial excess return (over cash) minus the excess
  return of buying and holding the same ETF (also net of its one entry cost); FX trial: excess
  return over cash (the benchmark is cash).
- **Primary statistic:** annualised Sharpe ratio of active returns (information ratio).
- Warm-up: signals may use data from before a period starts; returns are counted only inside
  the period. Each trial's first 252 trading days are excluded from evaluation in the
  development period, so all rules are judged on the same dates per asset.

## 6. Tests (development period, then once on the holdout)
| ID | Question | Method | Decision rule |
|---|---|---|---|
| H1 | Does any trial beat its benchmark, allowing for the search? | Hansen SPA (consistent p-value) on active returns; ETFs pooled, FX pooled, all pooled; stationary bootstrap, 1,000 reps, seed 2026 | reject "no trial beats the benchmark" at p < 0.05 |
| H2 | Which trials survive? | Romano-Wolf StepM (FWER 5%) and BHY haircut Sharpe (FDR 5%) on active returns | survivors listed |
| H3 | Is the best trial's Sharpe more than luck? | Deflated Sharpe on active returns; N = effective trials (participation ratio on active returns); V[SR] across all trials of that asset class | DSR >= 0.95 |
| H4 | Does picking the best rule in sample work out of sample? | PBO by CSCV (S = 16) per asset over that asset's trials | report the median PBO across assets and the share above 0.5 |
| H5 | Would a researcher who kept re-picking the best rule so far beat the benchmark? | Walk-forward per asset: expanding window, first 5 years training, then 1-year test steps; CPCV (6 groups, 2 test groups) path Sharpe ratios | out-of-sample active Sharpe per asset; share of assets above 0 |
| H6 | How much do costs matter? | Cost sweep on all trials | share of trials with positive active Sharpe at 0x, 1x, 2x, 3x costs |

**Holdout (opened once, after every development result is committed):** the H2 survivors (or,
if there are none, the 10 trials with the highest development active Sharpe) and the
walk-forward selections are run on the holdout. Results are committed unchanged, whatever
they show, in a commit tagged `pillar4-holdout-opened`.

## 7. What will be published
README section generated from the results (no hand-typed numbers), the full trial table
(parameters and statistics, no vendor data), charts in light and dark, and this document. Any
deviation from this plan is listed in a dated "Deviations" section below, with the reason.

## Deviations

### 2026-11-26: clarifications made before the six tests were run
These settle details the plan above left open. They were written before any of H1 to H6 was
computed. Before writing them, a smoke test of the pipeline had shown the per-trial active
Sharpe ratios of a few SPY trials and the net asset values of one example of each rule; no
rule, grid or test was changed after that.

1. **Start dates.** "First date with prices and rates" means the later of the two. Currencies
   with long FX histories start where their short rate starts (GBP 1997, AUD 1990, NZD 1973,
   JPY 1985, CHF 1972, CAD 1975, NOK and SEK 1982; EUR 1999).
2. **Pooled tests need common dates.** H1 (SPA), the StepM part of H2 and the effective number
   of trials in H3 need every trial's returns on the same days. They use the dates common to
   every asset in the pool after its warm-up: ETFs from the end of CPER's warm-up (late 2012),
   currencies from the end of EUR's (early 2000), all assets together from late 2012. Per-asset
   tests (H4, H5) and per-trial statistics use each asset's full evaluation period.
3. **BHY family.** The BHY haircut in H2 treats all 2,364 trials as one family; each trial's
   p-value uses its own sample length.
4. **DSR inputs.** In H3 the variance of Sharpe ratios is taken across all trials of the asset
   class, each on its own evaluation period, using per-day Sharpe ratios.
5. **Walk-forward length.** "First 5 years" in H5 means the first 1,260 trading days of the
   asset's evaluation period; each test step is 252 trading days.

### 2026-11-26: additions made after seeing the development results
These add information; they do not change any pre-registered test or decision rule.

1. **DSR with the raw trial count.** The effective number of trials (participation ratio)
   counted 2,013 ETF trials as about 12 and 351 currency trials as about 15, which is generous
   to the best trial. The DSR is also reported with N = the raw trial count, labelled as a
   sensitivity check.
2. **Buy-and-hold per asset.** The per-asset table adds the buy-and-hold CAGR over the
   evaluation period and the share of trials beating the benchmark, to show whether positive
   active returns come from timing or from avoiding an asset that lost money.
