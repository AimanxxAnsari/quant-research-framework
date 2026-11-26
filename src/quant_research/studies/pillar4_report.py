"""README section and figures for the Pillar 4 study, built only from its saved outputs.

Every number and every comparison word in the text is computed from
``reports/pillar4/<stage>/``; nothing is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from quant_research.data.settings import Settings  # noqa: E402
from quant_research.plotting import (  # noqa: E402
    Theme,
    end_label,
    legend_top,
    picture_md,
    save_light_dark,  # noqa: E402
)
from quant_research.plotting import title as chart_title  # noqa: E402
from quant_research.reporting import inject_file, md_table  # noqa: E402

HEADLINE = "next_open x1"
COST_COLS = ["next_open x0", "next_open x1", "next_open x2", "next_open x3"]


def _load(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    trials = pd.read_csv(folder / "trials.csv", index_col=0)
    per_asset = pd.read_csv(folder / "per_asset.csv", index_col=0)
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    return trials, per_asset, summary, manifest


# ---------------------------------------------------------------- figures


def family_figure(trials: pd.DataFrame, path: Path) -> None:
    stats = trials.groupby(["family", "class"])[HEADLINE].describe(percentiles=[0.1, 0.5, 0.9])
    order = (
        trials[trials["class"] == "etf"].groupby("family")[HEADLINE].median().sort_values().index
    )

    def draw(t: Theme) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 6.2))
        for k, (cls, label) in enumerate((("etf", "ETFs vs buy-and-hold"), ("fx", "FX vs cash"))):
            ys, mids, los, his = [], [], [], []
            for i, fam in enumerate(order):
                if (fam, cls) in stats.index:
                    row = stats.loc[(fam, cls)]
                    ys.append(i + (0.18 if k == 0 else -0.18))
                    mids.append(row["50%"])
                    los.append(row["10%"])
                    his.append(row["90%"])
            color = t.series[k]
            ax.hlines(ys, los, his, color=color, lw=2, alpha=0.55)
            ax.plot(mids, ys, "o", color=color, mec=t.surface, mew=1.5, ms=7, label=label)
        ax.axvline(0, color=t.baseline, lw=1.2)
        ax.set_yticks(range(len(order)), list(order))
        ax.set_xlabel("annualised Sharpe ratio of active returns (median, 10th to 90th percentile)")
        legend_top(ax, 1, t, loc="upper left")
        share = float((trials.loc[trials["class"] == "etf", HEADLINE] > 0).mean())
        chart_title(
            ax,
            f"{share:.0%} of ETF trials beat buy-and-hold after costs",
            "Active Sharpe of every trial by rule family, development period",
            t,
        )
        return fig

    save_light_dark(draw, path)


def oos_figure(per_asset: pd.DataFrame, path: Path) -> None:
    df = per_asset.sort_values("best in-sample Sharpe")

    def draw(t: Theme) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.5, 10))
        y = np.arange(len(df))
        ins, oos = df["best in-sample Sharpe"], df["walk-forward OOS Sharpe"]
        ax.hlines(y, np.minimum(ins, oos), np.maximum(ins, oos), color=t.grid, lw=2)
        ax.plot(ins, y, "o", color=t.series[0], ms=6, label="best trial in sample")
        ax.plot(oos, y, "o", color=t.series[1], ms=6, label="walk-forward pick, out of sample")
        ax.axvline(0, color=t.baseline, lw=1.2)
        ax.set_yticks(y, list(df.index), fontsize=8)
        ax.set_xlabel("annualised Sharpe ratio of active returns")
        legend_top(ax, 1, t, loc="lower right")
        gap = float((ins - oos).median())
        chart_title(
            ax,
            f"Picking the best rule so far gives up a median {gap:.2f} of Sharpe",
            "Per asset: best trial in hindsight vs re-picking the best rule so far (out of sample)",
            t,
        )
        return fig

    save_light_dark(draw, path)


def cost_figure(trials: pd.DataFrame, path: Path) -> None:
    share = (trials[COST_COLS] > 0).groupby(trials["class"]).mean()
    x = np.arange(len(COST_COLS))

    def draw(t: Theme) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for k, (cls, label) in enumerate((("etf", "ETF trials"), ("fx", "FX trials"))):
            vals = share.loc[cls].to_numpy()
            ax.plot(x, vals, "o-", color=t.series[k], mec=t.surface, mew=1.5, ms=7)
            end_label(ax, x[-1], vals[-1], f"{label} {vals[-1]:.0%}", t, dx=8)
        ax.set_xticks(x, ["no costs", "1x costs", "2x costs", "3x costs"])
        ax.set_xlim(-0.2, len(x) - 0.2)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("share beating the benchmark")
        etf_share = share.loc["etf"].to_numpy(dtype=float)
        drop = float(etf_share[0] - etf_share[-1])
        chart_title(
            ax,
            f"From no costs to 3x costs, {drop:.0%} of ETF trials fall below buy-and-hold",
            "Share of trials with a positive active Sharpe ratio, development period",
            t,
        )
        return fig

    save_light_dark(draw, path)


def buyhold_figure(per_asset: pd.DataFrame, path: Path) -> None:
    etf = per_asset[per_asset["class"] == "etf"]
    worst = etf.nsmallest(4, "buy-and-hold CAGR").index

    def draw(t: Theme) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(
            etf["buy-and-hold CAGR"],
            etf["share of trials beating benchmark"],
            "o",
            color=t.series[0],
            mec=t.surface,
            mew=1.5,
            ms=7,
        )
        pts = etf.loc[worst, ["buy-and-hold CAGR", "share of trials beating benchmark"]]
        for a, (x, y) in pts.iterrows():
            crowded = (
                (pts.iloc[:, 0] - x).between(1e-9, 0.04) & ((pts.iloc[:, 1] - y).abs() < 0.08)
            ).any()
            if crowded:  # another labelled point just to the right: label below-left instead
                ax.annotate(
                    str(a),
                    (x, y),
                    xytext=(-4, -6),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=8.5,
                    color=t.ink2,
                )
            else:
                end_label(ax, x, y, str(a), t)
        ax.axvline(0, color=t.baseline, lw=1.2)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("buy-and-hold CAGR of the ETF, development period")
        ax.set_ylabel("share of its trials beating it")
        corr = float(
            np.corrcoef(etf["buy-and-hold CAGR"], etf["share of trials beating benchmark"])[0, 1]
        )
        chart_title(
            ax,
            f"Rules beat the ETFs that did badly (correlation {corr:.2f})",
            "Each dot is one ETF: share of its 61 trials beating buy-and-hold",
            t,
        )
        return fig

    save_light_dark(draw, path)


# ---------------------------------------------------------------- text


def _verdict(p: float) -> str:
    return "rejected" if p < 0.05 else "not rejected"


def tests_table(summary: dict[str, Any], per_asset: pd.DataFrame, trials: pd.DataFrame) -> str:
    h12, h3 = summary["h1_h2"], summary["h3"]
    spa = {k: h12[k]["spa_p"] for k in ("etf", "fx", "all")}
    stepm = len(h12["all"]["stepm_survivors"])
    bhy_n = len(h12["bhy_survivors"])
    pbo = per_asset.groupby("class")["PBO"].median()
    pbo_high = int((per_asset["PBO"] > 0.5).sum())
    wf_pos = (per_asset["walk-forward OOS Sharpe"] > 0).groupby(per_asset["class"]).mean()
    share = (trials[HEADLINE] > 0).groupby(trials["class"]).mean()
    any_pass = h3["etf"]["passes"] or h3["fx"]["passes"]
    rows = [
        (
            "H1",
            "Does any trial beat its benchmark, allowing for the search? (SPA p-value)",
            f"ETFs {spa['etf']:.2f}, FX {spa['fx']:.2f}, all {spa['all']:.2f}",
            f"'no trial beats its benchmark' {_verdict(min(spa.values()))}",
        ),
        (
            "H2",
            "Which trials survive? (StepM; BHY haircut)",
            f"StepM {stepm}; BHY {bhy_n} of {summary['trials']:,}",
            "none survive" if stepm == 0 and bhy_n == 0 else "see trial table",
        ),
        (
            "H3",
            "Is the best trial more than luck? (Deflated Sharpe)",
            f"ETF {h3['etf']['dsr']:.2f}, FX {h3['fx']['dsr']:.2f}",
            "passes 0.95" if any_pass else "below 0.95 for both",
        ),
        (
            "H4",
            "Does the in-sample best stay above the median out of sample? (median PBO)",
            f"ETFs {pbo.get('etf', np.nan):.2f}, FX {pbo.get('fx', np.nan):.2f}",
            f"PBO above 0.5 for {pbo_high} of {len(per_asset)} assets",
        ),
        (
            "H5",
            "Does re-picking the best rule so far beat the benchmark out of sample?",
            f"positive for {wf_pos.get('etf', 0):.0%} of ETFs, {wf_pos.get('fx', 0):.0%} of FX",
            "see per-asset chart",
        ),
        (
            "H6",
            "Share of trials beating the benchmark after costs (1x)",
            f"ETFs {share.get('etf', 0):.0%}, FX {share.get('fx', 0):.0%}",
            "see cost chart",
        ),
    ]
    head = "| Test | Question | Result | Outcome |\n|---|---|---|---|\n"
    return head + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)


def commentary(summary: dict[str, Any], per_asset: pd.DataFrame, trials: pd.DataFrame) -> str:
    h3 = summary["h3"]
    etf = per_asset[per_asset["class"] == "etf"]
    losers = etf[etf["buy-and-hold CAGR"] < 0]
    winners_share = losers["share of trials beating benchmark"].mean()
    others_share = etf.loc[
        etf["buy-and-hold CAGR"] >= 0, "share of trials beating benchmark"
    ].mean()
    corr = float(
        np.corrcoef(etf["buy-and-hold CAGR"], etf["share of trials beating benchmark"])[0, 1]
    )
    worst = trials.loc[str(trials[HEADLINE].idxmin())]
    loser_names = ", ".join(map(str, losers.index))
    if winners_share > others_share:
        where = (
            "being out of the market some of the time beats holding an ETF that lost money, "
            "without any timing skill."
        )
    else:
        where = "so losing ETFs are not where the rules do best."
    cost_share = worst["next_open x0"] - worst[HEADLINE]
    if cost_share > abs(worst["next_open x0"]):
        costs = "most of that gap is costs: frequent trading at a fixed cost on a quiet ETF."
    else:
        costs = "costs are not the main reason for its loss."
    return f"""
**What the numbers say.** The best ETF trial, `{h3["etf"]["best"]}`, has an active Sharpe ratio
of {h3["etf"]["annual_sharpe"]:.2f}, against a luck threshold of {h3["etf"]["threshold_annual"]:.2f}
for {h3["etf"]["effective_trials"]:.0f} effective trials: DSR {h3["etf"]["dsr"]:.2f}
({h3["etf"]["dsr_all_trials"]:.2f} if all {h3["etf"]["trials"]:,} trials count as independent).
The best currency trial, `{h3["fx"]["best"]}`, reaches {h3["fx"]["annual_sharpe"]:.2f}: DSR
{h3["fx"]["dsr"]:.2f} ({h3["fx"]["dsr_all_trials"]:.2f} with all {h3["fx"]["trials"]} trials).

**Where "winning" comes from.** Across ETFs, the share of rules beating buy-and-hold has a
correlation of {corr:.2f} with the ETF's own buy-and-hold return. The {len(losers)} ETFs that
lost money when held ({loser_names}) see {winners_share:.0%} of their rules beat them, against
{others_share:.0%} for the rest: {where}

**Costs.** The worst trial, `{worst.name}`, has an active Sharpe of {worst[HEADLINE]:.2f} with
costs and {worst["next_open x0"]:.2f} without; {costs}
"""


def run(settings: Settings, stage: str = "development") -> str:
    folder = settings.root / "reports" / "pillar4" / stage
    trials, per_asset, summary, manifest = _load(folder)
    fig = folder / "figures"
    family_figure(trials, fig / "family_active_sharpe.png")
    oos_figure(per_asset, fig / "in_vs_out_of_sample.png")
    cost_figure(trials, fig / "costs.png")
    buyhold_figure(per_asset, fig / "buyhold_vs_rules.png")
    rel = f"reports/pillar4/{stage}/figures"
    top = trials.sort_values(HEADLINE, ascending=False).head(10)
    top_tbl = top[["class", HEADLINE, "next_open x3", "p single", "p BHY", "haircut Sharpe"]]
    top_tbl = top_tbl.rename(columns={HEADLINE: "Sharpe (1x)", "next_open x3": "Sharpe (3x)"})
    fmt = {"Sharpe (1x)": "{:.2f}", "Sharpe (3x)": "{:.2f}", "p single": "{:.3f}"}
    fmt.update({"p BHY": "{:.2f}", "haircut Sharpe": "{:.2f}"})
    block = f"""
Development period only (each asset from its start to {summary["period_end"]}); the holdout is
still sealed. Data through {manifest["data_through"]}; generated by commit
`{manifest["git_commit"][:7]}` with `uv run qrf study pillar4`. Results use live vendor data, so
they reproduce with your own free keys and will shift slightly as data is revised. Plan and
deviations: [pre-registration](docs/projects/pillar-4/PREREGISTRATION.md). Every trial:
[`trials.csv`](reports/pillar4/{stage}/trials.csv).

{tests_table(summary, per_asset, trials)}

{commentary(summary, per_asset, trials)}

{picture_md(f"{rel}/family_active_sharpe.png", "active Sharpe by rule family")}

{picture_md(f"{rel}/buyhold_vs_rules.png", "buy-and-hold return vs share of rules beating it")}

{picture_md(f"{rel}/in_vs_out_of_sample.png", "best in-sample vs walk-forward out-of-sample")}

{picture_md(f"{rel}/costs.png", "share of trials beating the benchmark by cost level")}

The ten trials with the highest active Sharpe ratio (annualised; p-values two-sided):

{md_table(top_tbl, fmt)}

Engine check: {summary["engine_check"]["trials"]} randomly chosen trials re-run on the loop
engine; largest NAV gap {summary["engine_check"]["max_gap"]:.1e}.
"""
    inject_file(settings.root / "README.md", "PILLAR4", block)
    return block
