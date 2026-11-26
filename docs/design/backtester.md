# Design: backtester (Pillar 4)

**Status:** engines built 2026-11-26 (step 2 of 5) · Package: `quant_research.backtest`

## Purpose
Turn a strategy's target weights into a daily NAV path under realistic timing, costs and
interest, in two independent ways that must agree, so the overfitting tools of Pillar 4 (and
every later strategy pillar) rest on accounting that has been checked.

## Scope / non-goals
In scope: ETFs and FX spot, daily bars, next-open and next-close fills, fixed bps costs,
interest on cash and FX balances, late-listed assets.
Not yet: futures (Pillar 3 supplies contracts and rolls; the market container has room for a
multiplier), no-trade bands, borrow fees, intraday data, live trading.

## Components

| Module | Role |
|---|---|
| `config.py` | `BacktestConfig` (fill rule, initial NAV, leverage limit, look-ahead checks) and `CostModel`; loaded from `config/backtest.yaml` |
| `market.py` | `Market`: validated open, close, rates, asset classes and day counts on one calendar; `up_to(date)`; `accrual_factors()` |
| `data.py` | `load_market(settings, etfs, fx)` builds a `Market` from the warehouse |
| `strategy.py` | `Strategy` protocol (`name`, `target_weights(market)`); `FixedWeights`, `MovingAverageCrossover` |
| `targets.py` | Checks shared by both engines: alignment, missing values, unpriced assets, leverage |
| `weights_engine.py` | Return-space engine (fast) |
| `loop_engine.py` | Units-and-cash engine, one bar at a time (cannot look ahead) |
| `lookahead.py` | `check_no_lookahead`: recompute on truncated data and compare |
| `runner.py` | `run` (engine switch), `run_matrix` (engines x fills), `engine_agreement` |
| `summary.py` | Headline numbers and a side-by-side table |
| `result.py` | `BacktestResult`: nav, weights, turnover, costs |

## Data flow
`load_market` -> `Market` -> strategy -> target weights (decided at each close) -> engine ->
`BacktestResult` -> `summary` / `engine_agreement`.

## Timeline of one day (both engines)
1. Interest from the previous date: cash x (1 + r_usd x days / 360); each FX balance x
   (1 + r_ccy x days / basis).
2. `next_open`: trade yesterday's decision at today's open.
3. `next_close`: trade yesterday's decision at today's close.
4. Mark NAV at today's close.
5. The strategy decides today's target from data up to today's close.

## Interfaces
```python
market = load_market(settings, etfs=["SPY", "TLT"], fx=["EUR", "JPY"])
config = load_config(settings)                       # or BacktestConfig(...)
result = run(MovingAverageCrossover("SPY", 50, 200), market, config, engine="weights")
results = run_matrix(strategy, market, config)       # both engines x both fills
side_by_side(results, market); engine_agreement(results)
```

## Invariants
- A decision at close t never trades at close t.
- Engines agree to rounding error on any valid input (tested at 1e-11 relative NAV).
- With zero costs and no borrowing, NAV matches vectorbt to 1e-12.
- Any non-zero target on an unpriced asset, any NaN target, leverage above the limit, and a
  NAV at or below zero raise specific errors.
- A strategy whose past decisions change when later data is removed raises `LookAheadError`
  in the weights engine.

## Edge cases
- Late-listed assets: missing prices before listing; must be held at zero until then.
- Missing FX fixing on a Fed holiday: carried up to 3 trading days.
- Weekends and holidays: interest uses calendar days.
- Short positions: proceeds earn the cash rate; no borrow fee.
- Wipe-out: error rather than a negative NAV.

## Test plan (all offline, synthetic)
Known answers for both engines and both fills: all cash earns the cash rate; buy and hold
earns the move from the fill price; a round trip costs twice the one-way cost; long FX earns
the foreign rate and short FX pays it; long FX gains the spot move; a decision never earns the
move that triggered it; leverage wipe-out raises. Engines agree with costs, carry, moving
rates and late listing (three seeds, both fills). Look-ahead: planted bug caught; with the
check off it looks brilliant and the loop engine disagrees; legitimate strategies pass.
vectorbt cross-check. Market, config and loader validation.

## Performance (MacBook, 6,764 days, 4 assets, MA 50/200)
Weights engine 20 ms (35 ms with the look-ahead check on 20 dates); loop engine 5.0 s. The
loop engine is O(T^2) in the strategy call because each bar re-runs the strategy on the
history so far; it is a reference, not the workhorse.

## Open questions
- No-trade band to cut drift trades (needs the same rule in both engines).
- Borrow spread for negative cash and short ETF borrow fees.
- Futures with multipliers and rolls once Pillar 3 exists.
