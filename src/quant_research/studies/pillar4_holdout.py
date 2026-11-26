"""Open the Pillar 4 holdout once (PREREGISTRATION.md section 6, "Holdout").

1. Refuse unless the development results are committed and unchanged, and the holdout has
   never been opened (no ``reports/pillar4/holdout`` and no ``pillar4-holdout-opened`` tag).
2. Re-run every trial on the full history (headline variant). The part before the holdout
   must reproduce the committed development returns: proof that nothing leaks backwards.
3. Pre-registered evaluation on 2021-12-01 onwards:
   * the H2 survivors, or if there are none, the 10 trials with the highest development active
     Sharpe ratio;
   * walk-forward per asset: at the start of the holdout and every 252 trading days after it,
     pick the trial with the best active Sharpe ratio on all data before that point, and hold
     it for the next 252 days (expanding window, as in H5).
4. Write ``reports/pillar4/holdout/``; the README section is made by ``pillar4_report``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_research.backtest.trial_log import run_manifest
from quant_research.data.settings import Settings
from quant_research.errors import ConfigError, DataIntegrityError
from quant_research.reporting import cell
from quant_research.studies.pillar4 import ASSETS, cache_dir, report_dir
from quant_research.studies.pillar4_grid import HOLDOUT_START
from quant_research.studies.pillar4_trials import (
    HEADLINE,
    AssetTrials,
    annual_sharpe,
    load,
    run_all,
    save,
)
from quant_research.validation.selection import select_out_of_sample
from quant_research.validation.splits import Split

TAG = "pillar4-holdout-opened"
N_TOP = 10
STEP = 252
CONSISTENCY_TOL = 1e-12


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    ).stdout.strip()


def guard(settings: Settings) -> None:
    """Fail unless this is the first, clean opening of the holdout."""
    root = settings.root
    dev = report_dir(settings)
    if not _git(root, "ls-files", str(dev / "summary.json")):
        raise ConfigError("development results must be committed before opening the holdout")
    if _git(root, "status", "--porcelain", "--", str(dev), "src", "config"):
        raise ConfigError("development results, code or config have uncommitted changes")
    if report_dir(settings, "holdout").exists() or _git(root, "tag", "-l", TAG):
        raise ConfigError("the holdout has already been opened; it is opened only once")


def holdout_splits(index: pd.DatetimeIndex, start: str, step: int = STEP) -> list[Split]:
    """Expanding-window steps that begin at the first date on or after ``start``."""
    first = int(index.searchsorted(pd.Timestamp(start)))
    if first == 0 or first >= len(index):
        raise ConfigError("the holdout start must fall inside the data, after some history")
    out = []
    for s in range(first, len(index), step):
        out.append(Split(np.arange(0, s), np.arange(s, min(s + step, len(index)))))
    return out


def check_consistency(full: list[AssetTrials], dev: list[AssetTrials]) -> float:
    """Largest difference between development returns and the same dates in the full run."""
    worst = 0.0
    for f, d in zip(full, dev, strict=True):
        same = f.active.loc[d.active.index, d.active.columns]
        worst = max(worst, float(np.nanmax(np.abs(same.to_numpy() - d.active.to_numpy()))))
    if worst > CONSISTENCY_TOL:
        raise DataIntegrityError(
            f"full-history returns differ from the committed development returns by {worst:.2e}"
        )
    return worst


def _selected_trials(settings: Settings) -> tuple[list[str], str]:
    dev = report_dir(settings)
    summary = json.loads((dev / "summary.json").read_text())
    survivors = sorted(
        set(summary["h1_h2"]["bhy_survivors"]) | set(summary["h1_h2"]["all"]["stepm_survivors"])
    )
    if survivors:
        return survivors, "H2 survivors"
    trials = pd.read_csv(dev / "trials.csv", index_col=0)
    top = trials[HEADLINE].sort_values(ascending=False).head(N_TOP).index.tolist()
    return [str(t) for t in top], f"no H2 survivors, so the {N_TOP} best development trials"


def top_trials(full: list[AssetTrials], ids: list[str], dev_trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid in ids:
        series = next(r.active[tid] for r in full if tid in r.active.columns)
        hold = series.loc[HOLDOUT_START:]
        rows.append(
            {
                "trial": tid,
                "development Sharpe": cell(dev_trials, tid, HEADLINE),
                "holdout Sharpe": annual_sharpe(hold.to_numpy()),
                "holdout days": len(hold),
            }
        )
    return pd.DataFrame(rows).set_index("trial")


def walk_forward_holdout(full: list[AssetTrials]) -> pd.DataFrame:
    rows = []
    for r in full:
        splits = holdout_splits(pd.DatetimeIndex(r.active.index), HOLDOUT_START)
        sel = select_out_of_sample(r.active.to_numpy(), splits)
        oos = sel.oos[~np.isnan(sel.oos)]
        picks = [str(r.active.columns[i]).split("|", 1)[1] for i in sel.chosen]
        rows.append(
            {
                "asset": r.asset,
                "class": r.asset_class,
                "holdout Sharpe": annual_sharpe(oos),
                "steps": len(splits),
                "picks": " ; ".join(picks),
            }
        )
    return pd.DataFrame(rows).set_index("asset")


def run_holdout(settings: Settings) -> dict[str, Any]:
    guard(settings)
    manifest = run_manifest(settings, {"stage": "holdout", "holdout_start": HOLDOUT_START})
    full = run_all(ASSETS, end=None, variants=(HEADLINE,))
    save(full, cache_dir(settings, "full"))
    gap = check_consistency(full, load(ASSETS, cache_dir(settings)))
    ids, rule = _selected_trials(settings)
    dev_trials = pd.read_csv(report_dir(settings) / "trials.csv", index_col=0)
    top = top_trials(full, ids, dev_trials)
    wf = walk_forward_holdout(full)
    end = max(r.active.index[-1] for r in full)
    summary = {
        "holdout_start": HOLDOUT_START,
        "holdout_end": str(end.date()),
        "selection": rule,
        "consistency_max_gap": gap,
        "top_trials_positive": int((top["holdout Sharpe"] > 0).sum()),
        "top_trials": len(top),
        "walk_forward_positive": {
            c: float((g["holdout Sharpe"] > 0).mean()) for c, g in wf.groupby("class")
        },
        "walk_forward_median": {
            c: float(g["holdout Sharpe"].median()) for c, g in wf.groupby("class")
        },
    }
    out = report_dir(settings, "holdout")
    out.mkdir(parents=True)
    top.to_csv(out / "top_trials.csv")
    wf.to_csv(out / "walk_forward.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary
