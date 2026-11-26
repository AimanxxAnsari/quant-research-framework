# Pillar 4: backtesting engine, validation toolkit and the popular-rules study

## Question
Do widely shared technical trading rules beat a passive benchmark once timing, costs and the
number of rules tried are taken into account? And can a one-person research stack answer that
honestly?

## Why it matters
Every systematic desk sees backtests that look good because many variants were tried.
Model-validation teams and portfolio managers need to separate skill from search: the
Deflated Sharpe Ratio, PBO and data-snooping tests are the standard tools, and a backtester
whose timing and costs are trustworthy is the prerequisite.

## What was built
1. **Two engines** on one set of conventions (ADR-0009, ADR-0010): a fast weights engine and a
   bar-by-bar loop engine that cannot look ahead. They agree to about 1e-14 on real data and
   match vectorbt where its conventions coincide.
2. **Look-ahead check**: every strategy is re-run on truncated data; a planted peek is caught.
3. **Validation toolkit** (ADR-0011): PSR, Deflated Sharpe with effective trials, minimum track
   record, PBO by CSCV, walk-forward, purged k-fold, CPCV, Harvey-Liu haircuts, Holm, BHY, SPA
   and StepM. Each reproduces a published example or a known answer.
4. **The study** (ADR-0012): 14 rule families, 2,364 trials on 33 ETFs and 9 currencies,
   pre-registered and tagged before any result, with a sealed five-year holdout.

## Results
All numbers are generated: see the **Pillar 4** section of the [README](../../../README.md)
and [`trials.csv`](../../../reports/pillar4/development/trials.csv). In short, for the
development period: no trial beats its benchmark after allowing for the search (SPA, StepM,
BHY), neither class's best trial clears the Deflated Sharpe bar, and the rules that "beat"
buy-and-hold on ETFs are mostly the ones applied to ETFs that lost money.

The holdout (December 2021 to November 2026) was opened once and published unchanged (tag
`pillar4-holdout-opened`); re-running on the full history reproduced the development returns
exactly. The ten best development trials were all on UNG, a decaying natural-gas ETF, and most
stayed ahead of holding it: consistent with "avoid the losing ETF", not with timing skill.
Re-picking the best rule each year beat the benchmark for only a minority of ETFs (see the
README's Holdout section for the generated numbers).

## What went wrong and how I found it
- A first planted look-ahead bug earned almost nothing, because the one-day fill lag moved the
  position past the return it peeked at. The fix made the test stronger, and it is a useful
  lesson: the most naive peek is defeated by realistic timing, so subtler ones matter more.
- vectorbt disagreed on leveraged portfolios: it never lets cash go negative. Found by reading
  its order records, not by guessing.
- The embargo in purged cross-validation started inside the purged rows and did nothing.
- A 2024 backtest failed on a 2019 gap in rate history it never used; rates now start at the
  backtest's start.
- The first full run's manifest showed uncommitted code (the report module was being written at
  the same time). The study was re-run from a clean commit, and the manifest now lists any
  uncommitted files instead of a yes/no flag.

## Limitations
- Fixed costs per asset class, swept from 0x to 3x, not measured spreads.
- No borrow spread on negative cash, no ETF borrow fees (ETFs are long/flat anyway).
- FX has one noon fixing a day, so FX rules use closes only and trade at a different time of day
  from ETFs.
- The effective number of trials (participation ratio) is generous to the best trial; the DSR
  with the raw trial count is reported alongside.
- Survivorship: the ETFs are ones that still exist today.

## Next steps
- Pillar 6 and 7 strategies run through the same engines, trial log and tests.
- A no-trade band, a borrow spread and measured spreads, each added to both engines.
