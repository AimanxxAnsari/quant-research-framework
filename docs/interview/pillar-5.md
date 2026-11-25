# Interview pack: synthetic data and known-answer tests

## 30-second pitch
"Before trusting any method on real data I measured how often it lies on simulated data where I
know the answer. Unrelated random walks look 'significant' in 90% of naive regressions; one year
of data underestimates a 10-day half-life by about 20%; screening 190 random pairs finds about
10 fake cointegrated pairs; a normal VaR on fat tails fails its backtest every time; and the best
of 100 no-skill strategies shows a 10-year Sharpe of 0.78, exactly where theory puts it."

## Likely questions
1. **Why is the AR(1) half-life biased?** OLS estimates of persistence are biased downward in
   small samples (Kendall: about -(1+3 phi)/n), so reversion looks faster.
2. **What is a spurious regression?** Two independent random walks regressed in levels give high
   R^2 and large t-statistics because the residuals are non-stationary; the t-test is invalid.
3. **Why does Engle-Granger work where the t-test fails?** It tests the residual for a unit root
   with critical values derived for that situation (MacKinnon).
4. **Why did the homoskedastic VR test over-reject on GARCH data?** Its variance formula assumes
   constant variance; clustering inflates the true variance of the statistic.
5. **Why does a normal VaR fail on t-distributed returns?** The 99% normal quantile is below the
   t quantile, so losses exceed it more often than 1%.
6. **How does the expected maximum Sharpe formula work?** Expected maximum of N standard normals
   scaled by the Sharpe estimate's standard error (about 1/sqrt(years)).
7. **Why was the MA grid's best in-sample Sharpe below the formula?** The 25 rules are highly
   correlated, so the effective number of independent trials is smaller.
8. **How do you test for look-ahead?** Plant same-day information that only a buggy
   implementation could exploit and assert it earns nothing.

## Must be able to rebuild from scratch
- [ ] Exact OU / AR(1) simulation from a half-life (phi = 0.5^(1/h)).
- [ ] GARCH(1,1) simulation.
- [ ] Kupiec LR statistic.
- [ ] Expected max Sharpe formula.
- [ ] MA crossover with correct timing.

## Numbers to remember
Naive t on unrelated random walks: 90% false positives. Half-life 10 -> median 8.2 with 250 days.
VR z under GARCH: 25% rejections; z*: 5.6%. Normal VaR on t-tails: 1.9% breaches.
Best of 100 noise strategies, 10 years: Sharpe 0.78 (theory 0.80).
