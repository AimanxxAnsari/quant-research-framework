# Volatility estimators, non-synchronous data and random-walk tests

## Range-based volatility (daily OHLC, log prices)
- Parkinson: s2 = (ln H/L)^2 / (4 ln 2)
- Garman-Klass: s2 = 0.5 (ln H/L)^2 - (2 ln 2 - 1)(ln C/O)^2
- Rogers-Satchell: s2 = ln(H/C) ln(H/O) + ln(L/C) ln(L/O)   (robust to drift)
- Yang-Zhang: s2 = var(overnight) + k var(open-to-close) + (1 - k) RS,
  k = 0.34 / (1.34 + (n + 1)/(n - 1))   (adds overnight gaps)
Average each over a window, multiply by 252, take the square root for annual vol.

## EWMA (RiskMetrics)
s2_t = lambda s2_{t-1} + (1 - lambda) r_{t-1}^2, lambda = 0.94 for daily data. The value used on
day t must only contain returns up to t-1.

## Non-synchronous data
If y is observed hours before x closes, part of today's common move appears in y tomorrow.
Daily corr(x_t, y_t) is too low and corr(x_t, y_{t+1}) is positive. Fixes: sum cross-covariances
over lags -1, 0, +1 (Scholes-Williams 1977; Dimson 1979), or use weekly data.

## Variance ratio (Lo & MacKinlay 1988)
VR(q) = Var(r_t + ... + r_{t-q+1}) / (q Var(r_t)). Random walk: 1. VR(2) = 1 + rho_1.
Robust statistic z* uses weights delta_j built from products of squared demeaned returns, so it
remains valid when volatility clusters.

## Hurst exponent
Split returns into blocks of size n; in each, R = range of the cumulative demeaned sum, S = std;
average R/S grows like n^H. H = 0.5 random walk. R/S is biased up in small samples and by short
memory. Lo's (1991) modified R/S replaces S with a long-run standard deviation (Newey-West style,
bandwidth q); under no long memory, V lies in [0.809, 1.862] with 95% probability.
