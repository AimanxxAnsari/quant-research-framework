# Known-answer testing

## OU from a half-life
Discrete OU = AR(1): x_t = m + phi (x_{t-1} - m) + e_t. Deviations shrink by phi each step, so they
halve after h steps when phi^h = 1/2, i.e. phi = 0.5^(1/h) and h = ln(0.5) / ln(phi).

## Size and power
- **Size:** how often a test rejects when the null is true (should equal the nominal level, e.g. 5%).
- **Power:** how often it rejects when the null is false.
Measure both by simulation before trusting a test.

## Kupiec proportion-of-failures test
With T days and x VaR breaches at target rate p:
LR = -2 [ (T - x) ln(1 - p) + x ln p - (T - x) ln(1 - x/T) - x ln(x/T) ] ~ chi-square(1).

## Expected maximum Sharpe of N no-skill strategies
E[max] ~ sd * ((1 - g) Phi^{-1}(1 - 1/N) + g Phi^{-1}(1 - 1/(N e))), g = 0.5772,
sd ~ 1/sqrt(years) for an annualised Sharpe. A backtest is only interesting if it beats this.

## Planted look-ahead test
Give the strategy information it must not use (the same day's return) and check it earns nothing.
A bug that lets it see that information produces an absurd Sharpe, which the test catches.
