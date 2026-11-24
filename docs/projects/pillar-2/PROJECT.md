# Pillar 2: statistics toolkit and first cross-asset results

## Abstract
A tested statistics toolkit (returns, six volatility estimators, distribution tests, beta and
lead-lag dependence, variance-ratio and long-memory tests) applied to 26+ years of ETF, FX and
rates data. Results (regenerated in the README by `qrf study pillar2`): every series is fat-tailed
(36 to 117 times the normal frequency of 4-sigma days); range-based volatility estimators predict
next-month volatility better than close-to-close; FX rates fixed at noon New York correlate with
the *previous* day's 4 pm equity close, so daily correlations understate the link (AUD:
0.27 daily vs 0.47 corrected and weekly); after Holm correction across 44 variance-ratio tests
only one series departs from a random walk; and the stock-bond correlation flipped sign in the
last year.

## 1. Questions
1. How non-normal are daily returns across asset classes?
2. Which volatility estimator best tracks next-month realised volatility on SPY?
3. How much does sampling time (noon FX fixing vs 4 pm close) distort measured correlation?
4. Do weekly returns deviate from a random walk once multiple testing is controlled?
5. Is there evidence of long memory once short-range dependence is accounted for?

## 2. Methods
See [design](../../design/stats-toolkit.md) and conventions
([ADR-0005](../../decisions/ADR-0005-statistics-conventions.md)). Every estimator was first
validated on simulated data with a known answer.

## 3. Results
Generated tables and figures: [README, Pillar 2 results](../../../README.md#pillar-2-results).

Key findings at first run (data to 2026-11-20):
- **Fat tails:** excess kurtosis from about 2.5 (EUR) to 15 (EEM); SPY, GLD, USO and 10Y yield
  changes negatively skewed.
- **Volatility estimators:** session-only estimators read about 2.5 vol points lower than
  close-to-close (they miss overnight gaps) yet correlate better with next-month realised vol;
  Yang-Zhang has the smallest forecast error.
- **Non-synchronous data:** next-day spillover is significant for AUD and EUR, absent for JPY; the
  lagged-covariance correction matches weekly correlation almost exactly. EWJ shows partial
  next-day reversal (Japanese market closed during US hours).
- **Variance ratios:** 3 of 44 tests significant after Holm, all for USD/JPY at longer horizons;
  the rest are consistent with random walks.
- **Long memory:** Lo's test rejects only for USO, which with 5 tests is weak evidence.
- **Correlation instability:** SPY-TLT weekly correlation -0.15 over the full sample, +0.27 over
  the last 12 months.

## 4. Limitations
- Daily data only; range estimators assume continuous trading.
- Variance-ratio and Hurst results depend on sample period; no sub-period robustness yet.
- ETF data begin in 2000 or later; FX and rates go back further, so samples differ by series.

## 5. Next
Pillar 4 (backtester with costs and overfitting controls) and Pillar 5 (synthetic known-answer
suite) build directly on these functions.
