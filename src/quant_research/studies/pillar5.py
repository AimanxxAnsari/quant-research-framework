"""Pillar 5 study: how often do standard quant methods fool you on data where the truth is
known? Pure simulation (no market data); fixed seeds; results generated into README.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from scipy import stats  # noqa: E402

from quant_research.data.settings import Settings  # noqa: E402
from quant_research.plotting import (  # noqa: E402
    Theme,
    end_label,
    legend_top,
    picture_md,
    ref_line,
    save_light_dark,
)
from quant_research.plotting import title as chart_title  # noqa: E402
from quant_research.reporting import cell, inject_file, md_table  # noqa: E402
from quant_research.risk.var import breaches, ewma_normal_var, historical_var, kupiec_pof  # noqa: E402
from quant_research.stats.mean_reversion import ar1_fit, engle_granger, ols_levels  # noqa: E402
from quant_research.stats.random_walk import variance_ratio  # noqa: E402
from quant_research.validation import overfitting as of  # noqa: E402
from quant_research.validation import synthetic as syn  # noqa: E402

ALPHA = 0.05


def _fig(out: Path, name: str) -> Path:
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out / "figures" / name


def half_life_study(out: Path, reps: int = 500, true_hl: float = 10.0) -> pd.DataFrame:
    rows, samples = {}, {}
    for n in (125, 250, 500, 1000, 2500):
        est = np.array(
            [
                ar1_fit(syn.ou(n, true_hl, 1.0, syn.rng(10_000 + s)))["half_life"]
                for s in range(reps)
            ]
        )
        samples[n] = np.clip(est, 0, 60)
        rows[f"{n} days"] = {
            "median estimate": float(np.median(est)),
            "5th pct": float(np.quantile(est, 0.05)),
            "95th pct": float(np.quantile(est, 0.95)),
            "share below truth": float(np.mean(est < true_hl)),
        }

    def draw(t: Theme):
        fig, ax = plt.subplots(figsize=(9, 3.8))
        xs = np.arange(len(samples))
        for x, est in zip(xs, samples.values(), strict=True):
            lo, med, hi = np.quantile(est, [0.05, 0.5, 0.95])
            q1, q3 = np.quantile(est, [0.25, 0.75])
            ax.plot([x, x], [lo, hi], color=t.series[0], lw=2, alpha=0.45)
            ax.plot([x, x], [q1, q3], color=t.series[0], lw=7, solid_capstyle="butt")
            ax.plot(x, med, "o", color=t.surface, mec=t.series[0], mew=2, ms=8)
            ax.annotate(
                f"{med:.1f}",
                (x, med),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=t.ink2,
            )
        ref_line(ax, y=true_hl, label=f"true half-life {true_hl:.0f} days", t=t)
        ax.set_xticks(xs, [f"{n:,}" for n in samples])
        ax.set_xlim(-0.5, len(samples) + 0.7)  # room for the reference label
        ax.set(xlabel="sample length (trading days)", ylabel="estimated half-life (days)")
        chart_title(
            ax,
            "Short samples make mean reversion look faster than it is",
            "Median (dot), 25-75% (bar) and 5-95% (line) of estimates; "
            f"{reps} simulations per length",
            t,
        )
        return fig

    save_light_dark(draw, _fig(out, "half_life_bias.png"))
    df = pd.DataFrame(rows).T
    df.index.name = f"sample (true half-life {true_hl:.0f})"
    return df


def cointegration_study(reps: int = 400, n: int = 500) -> dict:
    eg_null, naive_null, r2_null, power10, power50 = [], [], [], [], []
    for s in range(reps):
        w = syn.independent_random_walks(n, 2, syn.rng(20_000 + s))
        eg_null.append(engle_granger(w[:, 0], w[:, 1])["p_value"] < ALPHA)
        lv = ols_levels(w[:, 0], w[:, 1])
        naive_null.append(abs(lv["t"]) > 1.96)
        r2_null.append(lv["r2"])
        y, x = syn.cointegrated_pair(n, 1.5, 10, syn.rng(30_000 + s))
        power10.append(engle_granger(y, x)["p_value"] < ALPHA)
        y, x = syn.cointegrated_pair(n, 1.5, 50, syn.rng(40_000 + s))
        power50.append(engle_granger(y, x)["p_value"] < ALPHA)
    # Screening: 20 unrelated random walks -> 190 candidate pairs; how many look cointegrated?
    screens = []
    for s in range(20):
        w = syn.independent_random_walks(n, 20, syn.rng(50_000 + s))
        hits = sum(
            engle_granger(w[:, i], w[:, j])["p_value"] < ALPHA
            for i in range(20)
            for j in range(i + 1, 20)
        )
        screens.append(hits)
    return {
        "eg_false_positive": np.mean(eg_null),
        "naive_t_false_positive": np.mean(naive_null),
        "median_r2_unrelated": float(np.median(r2_null)),
        "power_hl10": np.mean(power10),
        "power_hl50": np.mean(power50),
        "screen_pairs": 190,
        "screen_hits_mean": float(np.mean(screens)),
    }


def variance_ratio_size_study(reps: int = 500, n: int = 1000) -> dict:
    out = {}
    for label, gen in [
        ("iid normal", lambda r: r.normal(0, 0.01, n)),
        ("GARCH(1,1)", lambda r: syn.garch11(n, r, alpha=0.15, beta=0.83, nu=5)[0]),
    ]:
        z, zs = [], []
        for s in range(reps):
            v = variance_ratio(gen(syn.rng(60_000 + s)), 2)
            z.append(abs(v.z) > 1.96)
            zs.append(abs(v.z_star) > 1.96)
        out[label] = {"homoskedastic z": float(np.mean(z)), "robust z*": float(np.mean(zs))}
    return out


def var_study(out: Path, reps: int = 100, n: int = 2500) -> pd.DataFrame:
    methods: dict[str, list[float]] = {
        "true conditional vol (t quantile)": [],
        "EWMA + normal quantile": [],
        "historical 250-day": [],
    }
    kupiec_rej: dict[str, list[bool]] = {k: [] for k in methods}
    nu = 5.0
    q_t = stats.t.ppf(0.99, nu) * np.sqrt((nu - 2) / nu)  # standardised-t 99% quantile
    for s in range(reps):
        r, sig = syn.garch11(n + 250, syn.rng(70_000 + s), nu=nu)
        forecasts = {
            "true conditional vol (t quantile)": np.r_[np.full(250, np.nan), q_t * sig[250:]],
            "EWMA + normal quantile": ewma_normal_var(r, 0.99),
            "historical 250-day": historical_var(r, 0.99, 250),
        }
        for k, v in forecasts.items():
            b = breaches(r, v)
            methods[k].append(b.mean())
            kupiec_rej[k].append(kupiec_pof(int(b.sum()), b.size)["p_value"] < ALPHA)
    df = pd.DataFrame(
        {
            "mean breach rate (target 1%)": {k: float(np.mean(v)) for k, v in methods.items()},
            "Kupiec rejects at 5%": {k: float(np.mean(v)) for k, v in kupiec_rej.items()},
        }
    )
    df.index.name = "99% VaR method on GARCH-t data"
    rates = df["mean breach rate (target 1%)"]

    def draw(t: Theme):
        fig, ax = plt.subplots(figsize=(9, 3.0))
        y = np.arange(len(rates))[::-1]
        values = rates.to_numpy(dtype=float)
        ax.barh(y, values, height=0.42, color=t.series[0])
        for yi, v in zip(y, values, strict=True):
            ax.annotate(
                f"{v:.1%}",
                (v, yi),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=t.ink,
            )
        ax.set_yticks(y, rates.index)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        ax.set_xlim(0, rates.max() * 1.25)
        ref_line(ax, x=0.01, label="target 1%", t=t)
        ax.set_xlabel("share of days the loss exceeded the 99% VaR")
        chart_title(
            ax,
            "99% VaR breach rates on fat-tailed, clustered returns",
            f"GARCH(1,1) with Student-t(5) shocks; {reps} paths of {n:,} days",
            t,
        )
        return fig

    save_light_dark(draw, _fig(out, "var_breach_rates.png"))
    return df


def overfitting_study(out: Path, reps: int = 300, ma_reps: int = 60) -> tuple[pd.DataFrame, dict]:
    """(a) N independent no-skill strategies over 10 years: best in-sample Sharpe vs theory.
    (b) A moving-average crossover grid on a random walk: best in-sample vs out-of-sample."""
    years, days = 10, 2520
    sd = 1 / np.sqrt(years)  # std of an annualised Sharpe estimate with no skill, ~1/sqrt(years)
    rows = {}
    curve_n = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    sim_curve = []
    r = syn.rng(80_000)
    for n_trials in curve_n:
        best = []
        for _ in range(reps if n_trials <= 200 else 60):
            daily = r.normal(0, 0.01, (days, n_trials))
            sr = daily.mean(axis=0) / daily.std(axis=0, ddof=1) * np.sqrt(252)
            best.append(sr.max())
        sim_curve.append(np.mean(best))
        if n_trials in (1, 10, 100, 1000):
            label = "strategy" if n_trials == 1 else "strategies"
            rows[f"{n_trials} {label}"] = {
                "best in-sample Sharpe (simulated)": float(np.mean(best)),
                "theory E[max]": of.expected_max_sharpe(n_trials, sd),
            }
    table = pd.DataFrame(rows).T
    table.index.name = "no-skill strategies, 10 years each"

    fast, slow = [5, 10, 20, 30, 50], [60, 100, 150, 200, 250]
    grid = [(f, s) for f in fast for s in slow]
    is_best, oos = [], []
    for s in range(ma_reps):
        lp = syn.gbm_log_prices(2 * days, 0.0, 0.16, syn.rng(90_000 + s))
        ins, outs = lp[: days + 1], lp[days:]
        srs = [of.annualised_sharpe(of.ma_crossover_returns(ins, f, sl)[sl:]) for f, sl in grid]
        k = int(np.argmax(srs))
        is_best.append(srs[k])
        f, sl = grid[k]
        oos.append(of.annualised_sharpe(of.ma_crossover_returns(outs, f, sl)[sl:]))
    ma = {
        "grid": len(grid),
        "best_is": float(np.mean(is_best)),
        "oos": float(np.mean(oos)),
        "theory_independent": of.expected_max_sharpe(len(grid), sd),
    }

    theory = [of.expected_max_sharpe(n, sd) for n in curve_n]

    def draw(t: Theme):
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(curve_n, theory, color=t.series[1], label="theory: expected maximum")
        ax.plot(
            curve_n,
            sim_curve,
            "o",
            color=t.series[0],
            mec=t.surface,
            mew=2,
            ms=8,
            label="simulated best Sharpe",
        )
        ax.set_xscale("log")
        ax.set_xticks(curve_n, [f"{n:,}" for n in curve_n])
        ax.minorticks_off()
        ax.axhline(0, color=t.baseline, lw=1)
        end_label(ax, curve_n[-1], sim_curve[-1], f"{sim_curve[-1]:.2f}", t, dx=8)
        ax.set(
            xlabel="number of no-skill strategies tried (log scale)",
            ylabel="best annualised Sharpe",
        )
        legend_top(ax, 1, t, loc="upper left")
        chart_title(
            ax,
            "Try enough strategies and noise looks like skill",
            "Best in-sample Sharpe among strategies with zero true edge, 10 years each",
            t,
        )
        return fig

    save_light_dark(draw, _fig(out, "max_sharpe_vs_trials.png"))
    return table, ma


def run(settings: Settings) -> str:
    out = settings.root / "reports" / "pillar5"
    hl = half_life_study(out)
    co = cointegration_study()
    vrs = variance_ratio_size_study()
    var = var_study(out)
    ofit, ma = overfitting_study(out)
    fig = "reports/pillar5/figures"
    pct = lambda v: f"{v:.1%}"  # noqa: E731

    summary = pd.DataFrame(
        [
            ("Engle-Granger test, unrelated random walks", "5.0%", pct(co["eg_false_positive"])),
            (
                "Naive levels t-test, unrelated random walks",
                "5.0%",
                pct(co["naive_t_false_positive"]),
            ),
            (
                "Variance ratio z (homoskedastic), GARCH data",
                "5.0%",
                pct(vrs["GARCH(1,1)"]["homoskedastic z"]),
            ),
            ("Variance ratio z* (robust), GARCH data", "5.0%", pct(vrs["GARCH(1,1)"]["robust z*"])),
            (
                "99% VaR, EWMA + normal, fat-tailed data",
                "1.0%",
                pct(var.loc["EWMA + normal quantile", "mean breach rate (target 1%)"]),
            ),
            (
                "Half-life estimate, 250 days, true 10",
                "10.0",
                f"{cell(hl, '250 days', 'median estimate'):.1f} (median)",
            ),
            (
                "Best of 100 no-skill strategies, 10y Sharpe",
                "0.00",
                f"{cell(ofit, '100 strategies', 'best in-sample Sharpe (simulated)'):.2f}",
            ),
        ],
        columns=["test or estimator", "truth / nominal", "measured"],
    ).set_index("test or estimator")

    block = f"""
Pure simulation with fixed seeds; regenerate with `uv run qrf study pillar5`. Every row below
uses data where the right answer is known.

### How often do standard methods fool you?

{md_table(summary, "{}")}

### Mean reversion: half-lives are underestimated in short samples

{
        md_table(
            hl,
            {
                "median estimate": "{:.1f}",
                "5th pct": "{:.1f}",
                "95th pct": "{:.1f}",
                "share below truth": "{:.0%}",
            },
        )
    }

{picture_md(f"{fig}/half_life_bias.png", "half-life")}

OLS on an AR(1) is biased towards faster mean reversion (Kendall 1954). With one year of data
the median estimate is {cell(hl, "250 days", "median estimate"):.1f} days for a true 10, and the
90% range spans {cell(hl, "250 days", "5th pct"):.1f} to {
        cell(hl, "250 days", "95th pct"):.1f} days.

### Cointegration: spurious relationships and screening

- Regressing one unrelated random walk on another: the naive t-test calls the slope significant
  in **{pct(co["naive_t_false_positive"])}** of cases (should be 5%), with median R-squared
  {co["median_r2_unrelated"]:.2f} (Granger & Newbold 1974).
- The Engle-Granger residual test does not over-reject: **{pct(co["eg_false_positive"])}** false
  positives at a 5% level.
- Power on a true pair (500 days, hedge ratio 1.5): **{pct(co["power_hl10"])}** with a 10-day
  half-life, **{pct(co["power_hl50"])}** with a 50-day half-life.
- Screening {co["screen_pairs"]} candidate pairs built from 20 unrelated random walks finds
  **{co["screen_hits_mean"]:.1f}** "cointegrated" pairs on average, all false. Any pairs screen
  must report how many pairs were tested ({co["screen_hits_mean"] / co["screen_pairs"]:.1%} of
  pairs pass at a 5% level, as expected when no relationship exists).

### Volatility clustering breaks naive test statistics

{md_table(pd.DataFrame(vrs).T.rename_axis("rejection rate at 5%, VR(2)"), "{:.1%}")}

On returns with GARCH volatility clustering the homoskedastic variance-ratio statistic rejects
a true random walk far too often; the robust z* (used in Pillar 2) stays close to 5%.

### 99% Value at Risk on fat-tailed, clustered returns

{md_table(var, "{:.1%}")}

{picture_md(f"{fig}/var_breach_rates.png", "var")}

Knowing the true volatility and tail shape gives the right breach rate. A normal quantile on
Student-t returns under-reserves, and a slow historical window reacts late to volatility
clusters. Pillar 9 extends this with conditional-coverage tests and Expected Shortfall.

### Backtest overfitting: the best of many no-skill strategies

{md_table(ofit, "{:.2f}")}

{picture_md(f"{fig}/max_sharpe_vs_trials.png", "max sharpe")}

Trying more strategies on noise mechanically raises the best Sharpe ratio, closely matching the
Bailey & Lopez de Prado formula. A grid of {ma["grid"]} moving-average crossover rules
on a driftless random walk produced a best in-sample Sharpe of **{ma["best_is"]:.2f}** and an
out-of-sample Sharpe of **{ma["oos"]:.2f}** for the selected rule. Because the rules are highly
correlated, the in-sample maximum is below the independent-trials benchmark
({ma["theory_independent"]:.2f}); the effective number of trials is smaller than the grid size.
Pillar 4 uses this benchmark in the Deflated Sharpe Ratio.
"""
    inject_file(settings.root / "README.md", "PILLAR5", block)
    return block
