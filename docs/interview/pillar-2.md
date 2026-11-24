# Interview pack: statistics toolkit

## 30-second pitch
"I built a tested stats toolkit and used it on 26 years of cross-asset data. Three findings I like:
range-based vol estimators read lower than close-to-close but predict next month's vol better;
FX rates fixed at noon correlate with the previous day's 4 pm equity close, so daily correlations
understate the link by almost half for AUD; and after correcting for 44 variance-ratio tests,
almost nothing rejects the random walk."

## Likely questions
1. **Log vs simple returns?** Log add across time; simple add across assets in a portfolio.
2. **Why are range estimators more efficient?** High and low carry information about the whole
   day's path, not just two points.
3. **Why do Parkinson / GK read lower than close-to-close?** They ignore overnight gaps;
   Yang-Zhang adds the overnight variance back.
4. **What is the non-synchronous trading problem?** Prices sampled at different times split a
   common move across two days; daily correlation is biased to zero; fix with lagged covariances
   (Scholes-Williams, Dimson) or lower frequency.
5. **What does VR(2) measure?** 1 + first-order autocorrelation; VR > 1 trending, < 1 reverting.
6. **Why z* instead of z?** Volatility clustering breaks the homoskedastic variance; z* is robust.
7. **Why Holm?** 44 tests at 5% would give about 2 false positives by chance; Holm controls the
   chance of any.
8. **Why not trust R/S Hurst?** Biased upward in finite samples and inflated by short-range
   autocorrelation; compare with shuffled data and use Lo's modified statistic.
9. **What does a stock-bond correlation flip mean for a 60/40 portfolio?** Bonds stop hedging
   equity drawdowns; diversification assumptions must be re-estimated.

## Must be able to rebuild from scratch
- [ ] Parkinson and Garman-Klass formulas.
- [ ] EWMA variance recursion without look-ahead.
- [ ] Variance ratio with overlapping returns.
- [ ] R/S statistic for one block.
- [ ] Lead-lag correlation.

## Numbers to remember
SPY excess kurtosis about 11.5; 4-sigma days 36-117x normal; AUD vs SPY 0.27 daily, 0.47 weekly;
3 of 44 VR tests significant after Holm; SPY-TLT -0.15 full sample vs +0.27 last year.
