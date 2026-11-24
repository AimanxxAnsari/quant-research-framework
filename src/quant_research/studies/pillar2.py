"""Pillar 2 study: return distributions, volatility estimators, non-synchronous data,
random-walk tests and cross-asset correlation on the warehouse data.

Writes figures to ``reports/pillar2/figures/`` and a Markdown block into README.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quant_research.data import access  # noqa: E402
from quant_research.data.settings import Settings  # noqa: E402
from quant_research.reporting import inject_file, md_table  # noqa: E402
from quant_research.stats import dependence as dep  # noqa: E402
from quant_research.stats import distribution  # noqa: E402
from quant_research.stats import random_walk as rw  # noqa: E402
from quant_research.stats import returns as ret  # noqa: E402
from quant_research.stats import volatility as vol  # noqa: E402

ETFS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "USO", "HYG"]
FX = {"DEXUSEU": "EUR", "DEXJPUS": "JPY", "DEXUSAL": "AUD"}


def _fig_path(out: Path, name: str) -> Path:
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out / "figures" / name


def distributions(settings: Settings) -> pd.DataFrame:
    px = access.prices(settings, ["SPY", "TLT", "GLD", "USO", "EEM"])
    fx = access.fx_usd_per_foreign(settings, ["DEXUSEU", "DEXJPUS"])
    y10 = access.series(settings, ["DGS10"])["DGS10"]
    rows = {t: distribution.describe(ret.log_returns(px[t])) for t in px}
    rows["EUR (USD per EUR)"] = distribution.describe(ret.log_returns(fx["DEXUSEU"]))
    rows["JPY (USD per JPY)"] = distribution.describe(ret.log_returns(fx["DEXJPUS"]))
    d = distribution.describe(y10.diff().iloc[1:] / 100)  # yield changes as decimals
    rows["UST 10Y yield change"] = d
    df = pd.DataFrame(rows).T
    out = pd.DataFrame(
        {
            "years": (df["n"] / 252).round(1),
            "ann. vol": df["vol_ann"],
            "skew": df["skew"],
            "excess kurtosis": df["excess_kurtosis"],
            "4-sd days per 100k": df["freq_beyond_4sd"] * 1e5,
            "normal: per 100k": df["normal_freq_beyond_4sd"] * 1e5,
        }
    )
    out.index.name = "series"
    return out


def vol_estimators(settings: Settings, out: Path) -> pd.DataFrame:
    bars = access.ohlc(settings, "SPY")
    est = vol.all_estimators(bars, window=21)
    fwd = vol.to_annual_vol(vol.close_to_close(bars["close"], 21)).shift(-21)
    rows = {}
    for name in est.columns:
        pair = pd.concat([est[name], fwd], axis=1).dropna()
        err = np.log(pair.iloc[:, 0] / pair.iloc[:, 1])
        rows[name] = {
            "mean ann. vol": float(est[name].mean()),
            "corr with next-month realised": float(pair.iloc[:, 0].corr(pair.iloc[:, 1])),
            "RMSE of log ratio": float(np.sqrt((err**2).mean())),
        }
    res = pd.DataFrame(rows).T
    res.index.name = "estimator (SPY, 21-day)"

    recent = est.loc["2019":, ["close_to_close", "parkinson", "yang_zhang", "ewma"]]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    recent.plot(ax=ax, lw=0.9)
    ax.set(title="SPY 21-day volatility estimators (annualised)", ylabel="vol", xlabel="")
    fig.tight_layout()
    fig.savefig(_fig_path(out, "spy_vol_estimators.png"), dpi=120)
    plt.close(fig)
    return res


def nonsynchronous(settings: Settings, out: Path) -> pd.DataFrame:
    spy = ret.log_returns(access.prices(settings, ["SPY", "EWJ"]))
    fx = ret.log_returns(access.fx_usd_per_foreign(settings, list(FX)))
    data = pd.concat([spy, fx], axis=1).dropna()
    weekly = pd.concat(
        [
            ret.log_returns(ret.resample_prices(access.prices(settings, ["SPY", "EWJ"]))),
            ret.log_returns(ret.resample_prices(access.fx_usd_per_foreign(settings, list(FX)))),
        ],
        axis=1,
    ).dropna()
    rows = {}
    for col, label in [
        ("EWJ", "EWJ (NY close, control)"),
        *[(k, f"{v} (noon NY fix)") for k, v in FX.items()],
    ]:
        ll = dep.lead_lag_corr(data["SPY"], data[col], range(-1, 2))
        rows[label] = {
            "daily corr": ll[0],
            "corr(SPY_t, X_t+1)": ll[1],
            "corr(SPY_t, X_t-1)": ll[-1],
            "lag-corrected daily": dep.sum_of_lags_corr(data["SPY"], data[col], 1),
            "weekly corr": float(weekly["SPY"].corr(weekly[col])),
            "days": float(len(data)),
        }
    res = pd.DataFrame(rows).T
    res.index.name = "SPY (4 pm close) vs"

    ll = dep.lead_lag_corr(data["SPY"], data["DEXUSAL"], range(-3, 4))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(ll.index, ll.values)
    ax.set(
        title="corr(SPY_t, AUD/USD_t+k): AUD fixed at noon, SPY closes at 4 pm",
        xlabel="lag k (days)",
        ylabel="correlation",
    )
    fig.tight_layout()
    fig.savefig(_fig_path(out, "spy_aud_lead_lag.png"), dpi=120)
    plt.close(fig)
    return res


def variance_ratios(settings: Settings) -> tuple[pd.DataFrame, int]:
    px = ret.resample_prices(access.prices(settings, ETFS))
    fx = ret.resample_prices(access.fx_usd_per_foreign(settings, list(FX)))
    rets = pd.concat([ret.log_returns(px), ret.log_returns(fx).rename(columns=FX)], axis=1)
    qs = [2, 4, 8, 16]
    records = []
    for name in rets:
        r = rets[name].dropna()
        for q in qs:
            v = rw.variance_ratio(r, q)
            records.append((name, q, v.vr, v.z_star, v.p_value))
    df = pd.DataFrame(records, columns=["series", "q", "vr", "z_star", "p"])
    # Holm correction across every test run.
    order = np.argsort(df["p"].to_numpy())
    m = len(df)
    adj = np.maximum.accumulate(df["p"].to_numpy()[order] * (m - np.arange(m)))
    df.loc[df.index[order], "p_holm"] = np.minimum(adj, 1)
    table = df.pivot(index="series", columns="q", values="vr").loc[[*ETFS, *FX.values()]]
    flags = df.pivot(index="series", columns="q", values="p_holm").loc[table.index]
    shown = table.copy().astype(object)
    for q in qs:
        shown[q] = [
            f"{v:.2f}" + (" (sig.)" if p < 0.05 else "")
            for v, p in zip(table[q], flags[q], strict=True)
        ]
    shown.columns = [f"VR({q})" for q in qs]
    shown.index.name = "weekly log returns"
    return shown, m


def hurst(settings: Settings) -> pd.DataFrame:
    px = access.prices(settings, ["SPY", "TLT", "GLD", "USO"])
    fx = access.fx_usd_per_foreign(settings, ["DEXUSEU"]).rename(columns={"DEXUSEU": "EUR"})
    rows = {}
    for name, p in [*px.items(), *fx.items()]:
        r = ret.log_returns(p).dropna()
        base, sd = rw.hurst_shuffled_baseline(r, n_shuffles=20)
        lo = rw.lo_modified_rs(r)
        rows[name] = {
            "R/S Hurst": rw.hurst_rs(r),
            "shuffled baseline": base,
            "baseline sd": sd,
            "Lo V": lo["V"],
            "Lo rejects no long memory": "yes" if lo["reject_no_long_memory"] else "no",
        }
    df = pd.DataFrame(rows).T
    df.index.name = "daily log returns"
    return df


def correlation_figure(settings: Settings, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = ["SPY", "EFA", "EEM", "HYG", "TLT", "IEF", "GLD", "USO", "UUP"]
    w = ret.log_returns(ret.resample_prices(access.prices(settings, tickers))).dropna()
    full = w.corr()
    last = w.loc[w.index.max() - pd.DateOffset(years=1) :].corr()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, c, title in [
        (axes[0], full, f"Weekly returns {w.index.min():%Y}-{w.index.max():%Y}"),
        (axes[1], last, "Last 12 months"),
    ]:
        im = ax.imshow(c.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(tickers)), tickers, rotation=45)
        ax.set_yticks(range(len(tickers)), tickers)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(_fig_path(out, "correlation_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return full, last


def run(settings: Settings) -> str:
    out = settings.root / "reports" / "pillar2"
    dist = distributions(settings)
    vols = vol_estimators(settings, out)
    nonsync = nonsynchronous(settings, out)
    vr, n_tests = variance_ratios(settings)
    hu = hurst(settings)
    corr_full, corr_last = correlation_figure(settings, out)
    obs = access.series(settings, ["DGS10"]).index.max()
    fig = "reports/pillar2/figures"

    # ----------------------------------------------------- data-driven commentary
    ratio = dist["4-sd days per 100k"] / dist["normal: per 100k"]
    neg_skew = [i for i in dist.index if dist.loc[i, "skew"] < -0.1]
    best = vols["corr with next-month realised"].idxmax()
    cc = vols.loc["close_to_close"]
    lowest_rmse = vols["RMSE of log ratio"].idxmin()
    lag_lines = []
    for name, r in nonsync.iterrows():
        c1, cutoff = r["corr(SPY_t, X_t+1)"], 3 / np.sqrt(r["days"])  # ~3 standard errors
        if c1 > cutoff:
            effect = "significant positive next-day spillover"
        elif c1 < -cutoff:
            effect = "significant negative next-day correlation, i.e. partial reversal"
        else:
            effect = "no detectable next-day effect"
        lag_lines.append(
            f"- **{name}:** daily {r['daily corr']:.2f}, next-day {r['corr(SPY_t, X_t+1)']:.2f}"
            f" ({effect}); lag-corrected {r['lag-corrected daily']:.2f} vs weekly "
            f"{r['weekly corr']:.2f}."
        )
    sig_cells = [(i, c) for i in vr.index for c in vr.columns if "(sig.)" in vr.loc[i, c]]
    sig_assets = sorted({i for i, _ in sig_cells})
    lo_rejects = [i for i in hu.index if hu.loc[i, "Lo rejects no long memory"] == "yes"]

    block = f"""
Data through {obs:%Y-%m-%d}. Regenerate with `uv run qrf study pillar2`.

### Fat tails everywhere

{
        md_table(
            dist,
            {
                "years": "{:.1f}",
                "ann. vol": "{:.1%}",
                "skew": "{:.2f}",
                "excess kurtosis": "{:.1f}",
                "4-sd days per 100k": "{:.0f}",
                "normal: per 100k": "{:.1f}",
            },
        )
    }

Every series has {ratio.min():.0f} to {ratio.max():.0f} times as many 4-standard-deviation days
as a normal distribution would produce. Negatively skewed: {", ".join(neg_skew) or "none"}.

### Volatility estimators

{md_table(vols)}

![vol estimators]({fig}/spy_vol_estimators.png)

Parkinson, Garman-Klass and Rogers-Satchell use only the trading session, so they read lower
than close-to-close ({cc["mean ann. vol"]:.1%}) by missing overnight gaps; Yang-Zhang adds the gap
back. As predictors of the next month's realised volatility, **{best}** has the highest
correlation and **{lowest_rmse}** the smallest error, versus {
        cc["corr with next-month realised"]:.2f}
correlation for close-to-close.

### Non-synchronous data: noon FX fixings vs a 4 pm equity close

{md_table(nonsync, {"days": "{:.0f}"})}

![lead-lag]({fig}/spy_aud_lead_lag.png)

FRED's H.10 FX rates are fixed at noon New York, four hours before SPY closes, so part of SPY's
afternoon move can only appear in the next day's fixing.

{chr(10).join(lag_lines)}

Where the spillover exists, summing lagged covariances recovers almost exactly the weekly
correlation, and daily correlation understates the link. EWJ trades in New York alongside SPY but
holds Japanese stocks whose market is closed then; its small negative next-day correlation is
consistent with partial reversal of US-session moves, so it is not a perfectly clean control.
Significance threshold: |corr| above 3/sqrt(days), about three standard errors.

### Random-walk tests: variance ratios (Lo & MacKinlay 1988)

{md_table(vr, "{}")}

VR(q) above 1 indicates trending, below 1 mean reversion. "(sig.)" marks ratios significant at 5%
using the heteroskedasticity-robust z* **after Holm correction across all {n_tests} tests**.
Significant after correction: {", ".join(sig_assets) or "none"}
({len(sig_cells)} of {n_tests} tests). Most weekly return series are statistically
indistinguishable from a random walk once the number of tests is accounted for.

### Long memory? Hurst exponent vs Lo's modified R/S

{md_table(hu)}

The classical R/S Hurst exponent is above 0.5 even on shuffled data (small-sample bias), so it is
read against that baseline. Lo's modified R/S statistic, robust to short-range dependence, rejects
"no long memory" (outside [0.809, 1.862]) for: {", ".join(lo_rejects) or "none"}
({len(lo_rejects)} of {len(hu)} tests at 5%, uncorrected). With {len(hu)} independent tests, the
chance of at least one false rejection is {1 - 0.95 ** len(hu):.0%}, so a single rejection is weak
evidence of long memory.

### Cross-asset correlation

![correlation]({fig}/correlation_matrix.png)

Stock-bond correlation (SPY vs TLT, weekly): {corr_full.loc["SPY", "TLT"]:.2f} over the full sample
vs {corr_last.loc["SPY", "TLT"]:.2f} in the last 12 months. Correlations are unstable; a hedge
that worked historically can stop working, which is why later pillars estimate them on rolling
windows and stress them.
"""
    inject_file(settings.root / "README.md", "PILLAR2", block)
    return block
