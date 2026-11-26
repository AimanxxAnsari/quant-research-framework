# ADR-0009: Backtest conventions

- **Date:** 2026-11-26
- **Status:** accepted

## Context
A backtest result depends on many small conventions: when a trade fills, which prices it uses,
how costs are charged, how idle cash and FX balances earn interest, and what happens when
calendars differ. Two engines (ADR-0010) must share exactly the same conventions, or their
disagreement means nothing. Choices for Pillar 4 agreed beforehand: ETFs and FX spot with
carry, next-open and next-close fills reported side by side, fixed bps costs per asset class
(Pillar 4 research notes, 2026-11-26).

## Decisions

| Topic | Choice | Rejected alternative and why |
|---|---|---|
| Strategy output | Target weights (signed fraction of NAV per asset) decided at each close; the rest is USD cash, which may be negative (borrowing) | Orders or share counts: harder to compare across assets and engines |
| Fill timing | A decision at close t fills at the open of t+1 (`next_open`) or the close of t+1 (`next_close`); both are run and reported | Fill at the close of t: assumes you can trade at the price you just used to decide, the classic look-ahead |
| Rebalancing | Trade back to the target at every fill, including small drift trades | Trade only when the target changes: path-dependent units make the weights engine sequential; a no-trade band can be added later as its own decision |
| Trade sizing | Units = target weight x NAV just before the trade / fill price; costs are then paid from cash | Sizing on post-cost NAV: needs an implicit solve, and differs by cost x weight, which is tiny |
| Costs | One-way, fixed bps of traded notional per asset class (ETF 2 bp, FX 1 bp by default), with a multiplier for 1x / 2x / 3x sweeps | Spread estimators from daily OHLC (Abdi-Ranaldo, EDGE): more realistic per ETF but add estimation noise; fixed costs are simple to state, and the sweep shows how much a result depends on them |
| ETF prices | Split- and dividend-adjusted open and close, so returns are total returns | Raw prices plus a dividend cash flow: same answer, more code |
| FX | A foreign-currency balance priced in USD per unit. FRED H.10 noon New York fixing used as both open and close | FX futures or forwards: no free data |
| Interest | USD cash and FX balances earn their short rate (ADR-0008) over calendar days to the next date, using the rate in force on the earlier date; negative balances pay the same rate | A borrow spread: not modelled yet; noted as a limitation |
| ETF shorts | No borrow fee; short proceeds sit in USD cash and earn the cash rate | Borrow fees: no free data; matters for hard-to-borrow names, not for these ETFs |
| Calendar | ETF trading days when any ETF is in the market, otherwise FX fixing days. A missing FX fixing is carried for at most 3 trading days | Union of all calendars: creates fake ETF prices on Fed-only business days |
| Late listing | Allowed: missing prices before an asset starts; any non-zero target on an unpriced asset is an error | Back-filling prices: invents data |
| Safety limits | Gross leverage above 4x raises an error; NAV at or below zero raises `AccountWipedOutError` | Carrying on with a negative NAV: produces meaningless returns |
| Headline numbers | Sharpe on returns in excess of USD cash, 252 days; CAGR, cost drag and turnover per calendar year | Sharpe on raw returns: an all-cash portfolio would show a positive Sharpe |

## Consequences
- Every test and study states its fill rule; results are reported for both.
- The FX fixing is at noon New York and the ETF close is at 4 pm, so a mixed ETF-FX strategy
  trades FX at a different time of day from ETFs. The next-open / next-close distinction only
  changes ETF fills.
- Daily rebalancing to fixed weights creates small drift trades (60/40 SPY/TLT: about 1.3x
  turnover a year, 3 bp a year of costs at the default cost).
