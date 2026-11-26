"""The six pre-registered tests (PREREGISTRATION.md section 6, with the dated clarifications)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from quant_research.reporting import cell
from quant_research.studies.pillar4_grid import SEED
from quant_research.studies.pillar4_trials import HEADLINE, TRADING_DAYS, VARIANTS, AssetTrials
from quant_research.validation.multiple_testing import (
    bhy,
    data_snooping,
    sharpe_from_pvalue,
    sharpe_pvalue,
)
from quant_research.validation.pbo import cscv
from quant_research.validation.selection import (
    column_sharpe,
    cpcv_out_of_sample,
    select_out_of_sample,
)
from quant_research.validation.sharpe import (
    deflated_sharpe,
    effective_number_of_trials,
    sharpe_stats,
)
from quant_research.validation.splits import walk_forward

ALPHA = 0.05
SPA_REPS = 1000
WF_TRAIN, WF_TEST = 1260, 252
PBO_BLOCKS = 16


def trial_table(results: Sequence[AssetTrials]) -> pd.DataFrame:
    """One row per trial: where, when, and its active Sharpe ratio under every variant."""
    rows = []
    for r in results:
        days = len(r.active)
        for tid in r.active.columns:
            asset, family, params = str(tid).split("|")
            row: dict[str, Any] = {
                "trial": tid,
                "asset": asset,
                "class": r.asset_class,
                "family": family,
                "params": params,
                "start": r.active.index[0].date(),
                "end": r.active.index[-1].date(),
                "days": days,
            }
            row.update({v: cell(r.sharpe, str(tid), v) for v in VARIANTS})
            rows.append(row)
    return pd.DataFrame(rows).set_index("trial")


def pooled(results: Sequence[AssetTrials], classes: set[str]) -> pd.DataFrame:
    """Active returns of every trial in the pool on the dates common to all of them."""
    frames = [r.active for r in results if r.asset_class in classes]
    return pd.concat(frames, axis=1, join="inner")


def h1_h2(results: Sequence[AssetTrials], table: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, classes in (("etf", {"etf"}), ("fx", {"fx"}), ("all", {"etf", "fx"})):
        m = pooled(results, classes)
        res = data_snooping(m.to_numpy(), size=ALPHA, reps=SPA_REPS, seed=SEED)
        out[label] = {
            "start": str(m.index[0].date()),
            "end": str(m.index[-1].date()),
            "trials": m.shape[1],
            "spa_p": res.spa_pvalue,
            "reality_check_p": res.spa_pvalue_upper,
            "stepm_survivors": [str(m.columns[i]) for i in res.better],
        }
    years = table["days"].to_numpy() / TRADING_DAYS
    sr = table[HEADLINE].to_numpy()
    p = np.array([sharpe_pvalue(s, y) for s, y in zip(sr, years, strict=True)])
    adj = bhy(p)
    cut = [sharpe_from_pvalue(a, y, s) for a, y, s in zip(adj, years, sr, strict=True)]
    table["p single"], table["p BHY"], table["haircut Sharpe"] = p, adj, cut
    table["BHY survivor"] = (adj < ALPHA) & (sr > 0)
    out["bhy_survivors"] = table.index[table["BHY survivor"]].tolist()
    return out


def h3(results: Sequence[AssetTrials], table: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in ("etf", "fx"):
        sub = table[table["class"] == cls]
        best = str(sub[HEADLINE].idxmax())
        series = next(r.active[best] for r in results if best in r.active.columns)
        st = sharpe_stats(series.to_numpy())
        per_day = sub[HEADLINE].to_numpy() / np.sqrt(TRADING_DAYS)
        n_eff = effective_number_of_trials(pooled(results, {cls}).to_numpy())
        d = deflated_sharpe(
            st.sharpe, st.n_obs, st.skew, st.kurtosis, float(np.var(per_day, ddof=1)), n_eff
        )
        out[cls] = {
            "best": best,
            "annual_sharpe": st.sharpe * np.sqrt(TRADING_DAYS),
            "trials": len(sub),
            "effective_trials": n_eff,
            "threshold_annual": d.threshold * np.sqrt(TRADING_DAYS),
            "dsr": d.dsr,
            "passes": d.dsr >= 0.95,
            # sensitivity added after seeing results (Deviations, 2026-11-26): raw trial count
            "dsr_all_trials": deflated_sharpe(
                st.sharpe, st.n_obs, st.skew, st.kurtosis, float(np.var(per_day, ddof=1)), len(sub)
            ).dsr,
        }
    return out


def h4_h5(results: Sequence[AssetTrials]) -> pd.DataFrame:
    rows = []
    for r in results:
        a = r.active.to_numpy()
        pbo = cscv(a, PBO_BLOCKS)
        wf = select_out_of_sample(a, walk_forward(len(a), WF_TRAIN, WF_TEST))
        oos = wf.oos[~np.isnan(wf.oos)]
        cp = cpcv_out_of_sample(a, 6, 2)
        rows.append(
            {
                "asset": r.asset,
                "class": r.asset_class,
                "PBO": pbo.pbo,
                "prob OOS loss": pbo.prob_oos_loss,
                "degradation slope": pbo.degradation_slope,
                "best in-sample Sharpe": float(column_sharpe(a).max() * np.sqrt(TRADING_DAYS)),
                "walk-forward OOS Sharpe": float(
                    column_sharpe(oos[:, None])[0] * np.sqrt(TRADING_DAYS)
                ),
                "CPCV median path Sharpe": float(
                    np.median(column_sharpe(cp.oos.T)) * np.sqrt(TRADING_DAYS)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("asset")


def h6(table: pd.DataFrame) -> pd.DataFrame:
    """Share of trials with a positive active Sharpe ratio under each variant, by class."""
    cols = ["next_open x0", HEADLINE, "next_open x2", "next_open x3", "next_close x1"]
    return (table[cols] > 0).groupby(table["class"]).mean()
