# Backtest accounting: two ways to get the same number

These are the two algorithms in `quant_research.backtest`, written so they can be rebuilt on a
whiteboard.

## The setting
- Each day t has an open O_t and a close C_t per asset. A strategy looks at data up to the close
  of day t and outputs target weights w_t (fraction of NAV per asset; the rest is cash).
- The decision is filled at the next trade point: the open or the close of day t+1.
- Cash grows by g_t = 1 + r_usd x days / 360 between dates; an FX balance grows by
  1 + r_ccy x days / basis on top of its price change.
- Cost of a trade = k x |traded notional|, with k = bps / 10,000.

## Way 1: units and cash (the loop engine)
State: cash, units u_i.
```
for each day t:
    cash *= g_t; u_i *= (FX interest factor)               # overnight interest
    if there is a pending target w:
        P = open_t or close_t                              # fill price
        nav = cash + sum(u_i * P_i)
        target_i = w_i * nav / P_i
        trade_i = target_i - u_i
        cost = sum(k_i * |trade_i| * P_i)
        cash -= sum(trade_i * P_i) + cost
        u = target
    NAV_t = cash + sum(u_i * C_t,i)
    w = strategy(data up to close t)                       # decided now, filled tomorrow
```
Because the strategy only ever receives data up to today, it cannot use the future.

## Way 2: weights and growth (the weights engine)
Let W_t be the weights filled at trade point t and q_t,i the growth of one dollar in asset i
between trade points t-1 and t (price ratio x FX interest factor). With X_t the NAV just
before the trade at t, and kappa_t the cost of that trade as a fraction of X_t:

  rho_t = X_t / X_{t-1} = sum_i W_{t-1,i} q_t,i + (1 - sum_i W_{t-1,i}) g_t - kappa_{t-1} g_t

  kappa_t = sum_i k_i | W_t,i - W_{t-1,i} q_t,i / rho_t |

The term W_{t-1,i} q_t,i / rho_t is the weight asset i has drifted to by the time of the trade;
the difference from the new target is what gets traded. The last term in rho is last trade's
cost, which was paid out of cash and so also missed a period of interest.

- **No costs:** kappa = 0, so NAV = X_0 x cumprod(rho), fully vectorised.
- **With costs:** rho_t needs kappa_{t-1}, and kappa_t needs rho_t, so it is a loop over time
  of scalars, with vector maths across assets at each step.
- **Next-open fills:** the trade points are opens; the reported close NAV multiplies each
  position by C_t / O_t on the day of the trade.

## Why they agree, and why that matters
Both follow the same events in the same order, but one counts units and the other compounds
weights. A bug in one (an off-by-one shift, a cost charged on the wrong NAV) shows up as a gap
between them. On 27 years of real data the gap is about 1e-14, which is rounding error.

## Look-ahead check
If the decision at date t is the same whether or not data after t exists, the strategy cannot
be using the future. The weights engine re-runs the strategy on data cut off at 20 dates and
compares. A classic planted bug, "be long when the return I am about to earn is positive",
fails this check at once.

## Things to be able to explain
- Why "fill at the same close you decided on" is look-ahead.
- Why a 1-day peek earns nothing with a 1-day fill lag, and a 2-day peek earns a fortune.
- Why vectorbt differs only when the portfolio needs negative cash (it does not borrow).
- Why daily rebalancing to fixed weights still trades (prices move, so weights drift).
