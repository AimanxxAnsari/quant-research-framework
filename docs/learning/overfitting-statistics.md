# Judging many backtests: the statistics

Written so each method can be rebuilt and explained without notes.

## Why the best backtest lies
Try N strategies with no skill and keep the best: its Sharpe ratio is the maximum of N noisy
numbers, so it is positive even though every true Sharpe is zero. Expected maximum of N
standard normals: (1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)), g = 0.5772.

## Probabilistic Sharpe Ratio (PSR)
The Sharpe estimate is noisy, more so with fat tails and negative skew:

  Var(SR_hat) ~ (1 - g3 SR + (g4 - 1)/4 SR^2) / (T - 1)

PSR(SR*) = Phi((SR - SR*) / sqrt(Var)). For normal returns (g3 = 0, g4 = 3) this is Lo's
(1 + SR^2 / 2) / (T - 1). Everything per period, kurtosis not excess.

## Deflated Sharpe Ratio (DSR)
PSR with SR* = the Sharpe the best of N no-skill trials reaches by luck:
SR0 = sqrt(V[SR across trials]) x expected max of N normals. The paper's example: annual SR 2.5
over 5 years, 100 trials, V = 1/2, skew -3, kurtosis 10 -> SR0 = 0.113 per day, DSR = 0.90: not
significant at 95%. With 46 trials it would have been 0.95.

## Minimum track record length
Solve PSR(SR*) = 0.95 for T: T = 1 + Var-factor x (z_0.95 / (SR - SR*))^2. Fat left tails make
it longer.

## Effective number of trials
Correlated trials are fewer bets. Participation ratio of the correlation matrix eigenvalues:
N^2 / sum(lambda^2). Independent -> N, identical -> 1, k identical clusters -> k.

## Probability of Backtest Overfitting (PBO) by CSCV
Cut the history into S blocks. For each of the C(S, S/2) ways to pick half as in-sample: choose
the best trial in-sample, rank it out-of-sample among all N, logit of rank / (N + 1). PBO =
share of splits where that logit <= 0 (the winner is at or below the median OOS). No skill:
about 0.5. Real skill: near 0. Also read: slope of OOS on IS Sharpe (negative = the more it
shines in-sample, the worse it does after).

## Cross-validation for time series
- Walk-forward: train on the past, test on the next window, step forward. One path.
- Purged k-fold: contiguous folds; drop training rows whose labels overlap the test block
  (purge), then a few more after it (embargo), so the test period cannot leak into training.
- CPCV: G groups, every choice of k as the test set: C(G, k) splits, k/G x C(G, k) complete
  paths. A distribution of out-of-sample Sharpe ratios instead of one number.

## Multiple testing
- t = SR_annual x sqrt(years); p two-sided.
- Bonferroni: p x N. Independent: 1 - (1 - p)^N. Holm: step-down Bonferroni (family-wise).
  BHY: false discovery rate under any dependence.
- Haircut Sharpe: turn the adjusted p back into a Sharpe ratio. Harvey and Liu's example:
  SR 0.75 over 20 years, 200 tests -> 0.32. Weak Sharpe ratios are cut hard, strong ones
  barely.

## White's Reality Check, Hansen's SPA, Romano-Wolf StepM
Bootstrap the whole set of trials together (stationary bootstrap keeps serial dependence) and
ask whether the best performance relative to a benchmark is larger than the search alone
would produce. SPA improves the Reality Check by not letting very bad trials dilute the test.
StepM lists which trials survive, controlling the family-wise error rate.

## Things to be able to explain
- Why selection plus noise produces a positive best Sharpe, and how DSR removes it.
- Why per-period Sharpe, and why kurtosis must not be excess in PSR.
- Why one dataset's PBO is noisy even with 12,870 splits.
- Why the benchmark choice decides what "significant" means.
