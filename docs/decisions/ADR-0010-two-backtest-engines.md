# ADR-0010: Two backtest engines, compared side by side

- **Date:** 2026-11-26
- **Status:** accepted

## Context
Fast vectorised backtests are easy to get subtly wrong (an off-by-one shift is a look-ahead
bug), and slow event-driven backtests are too slow for the hundreds of trials an overfitting
study needs. Yin et al. (2026) ran 15 strategies through five engines and found they agree at
zero cost but drift apart with costs, and found seven bugs in three engines. The plan for this
pillar is to build both kinds and compare them, with a switch between them.

## Decision
Two engines, one set of conventions (ADR-0009), one result type:

- **Weights engine** (`backtest.weights_engine`): asks the strategy once for all its target
  weights, then follows NAV in return space with weights and growth factors. Vectorised across
  assets; with costs, one scalar recursion over time. About 20 ms for 27 years of daily data.
- **Loop engine** (`backtest.loop_engine`): keeps USD cash and units of each asset, walks the
  dates one by one and asks the strategy for each decision with the market cut off at that
  day's close. It cannot look ahead. About 5 s for the same run.
- **Look-ahead check** (`backtest.lookahead`): the weights engine re-runs the strategy on data
  cut off at 20 evenly spaced dates and raises `LookAheadError` if any past decision changes.
- **Switch and comparison**: `runner.run(..., engine=...)` picks one; `runner.run_matrix` runs
  both engines under both fill rules; `engine_agreement` reports the largest NAV gap and
  `summary.side_by_side` the headline numbers. A Streamlit page with an engine toggle comes in
  step 5.
- **External check**: tests compare against vectorbt 1.2.1 (`from_orders`, target percent, same
  prices and fees).

## Test results at the time of writing
- Engines agree to a relative NAV gap below 1e-11 on random markets with costs, FX carry,
  moving rates and a late-listed asset, and to about 1e-14 on 27 years of SPY, TLT, EUR and JPY.
- vectorbt agrees to rounding error (tolerance 1e-12) with and without costs, as long as the
  portfolio never needs negative cash. vectorbt does not borrow: it shrinks orders that would
  need it, so leveraged or fully invested-plus-fees targets differ (found while writing the
  tests: targets of 119% and 29% were filled as 71% and 29%).
- A planted look-ahead (long when the return the position will earn is positive) is caught by
  the check; with the check switched off it turns 1 into more than 3 within one synthetic
  year, while the loop engine, which never sees the future, stays flat.

## Rejected
- **Only a vectorised engine:** fast, but nothing independent to catch its own bugs.
- **Only an event-driven engine:** too slow for the overfitting study (hundreds of trials).
- **Using vectorbt as the engine:** no FX carry or interest on cash, no borrowing, and less to
  learn and explain; kept as an external check instead.
- **Sharing one accounting core between the engines:** a bug in the core would show up in
  both and the comparison would not catch it. The engines share conventions and input
  checks, not arithmetic.

## Consequences
- Studies run on the weights engine; a sample of trials is re-run on the loop engine, and the
  gap is reported.
- Any new convention (a borrow spread, a no-trade band) must be added to both engines, with a
  test that they still agree.
