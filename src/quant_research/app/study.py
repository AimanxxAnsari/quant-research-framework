"""Read-only access to the committed Pillar 4 results for the study explorer page.

Only files already committed under ``reports/pillar4`` are read (per-trial statistics, no
prices), so the page works from a clean clone without any data or keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research.errors import DataNotFoundError


@dataclass(frozen=True)
class StudyResults:
    trials: pd.DataFrame
    per_asset: pd.DataFrame
    summary: dict[str, Any]
    manifest: dict[str, Any]
    holdout_summary: dict[str, Any] | None
    holdout_top: pd.DataFrame | None
    holdout_walk_forward: pd.DataFrame | None


def _need(path: Path) -> Path:
    if not path.exists():
        raise DataNotFoundError(f"{path} is missing; run `qrf study pillar4` first")
    return path


def load_study(root: Path) -> StudyResults:
    dev = root / "reports" / "pillar4" / "development"
    hold = root / "reports" / "pillar4" / "holdout"
    opened = (hold / "summary.json").exists()
    return StudyResults(
        trials=pd.read_csv(_need(dev / "trials.csv"), index_col=0),
        per_asset=pd.read_csv(_need(dev / "per_asset.csv"), index_col=0),
        summary=json.loads(_need(dev / "summary.json").read_text()),
        manifest=json.loads(_need(dev / "manifest.json").read_text()),
        holdout_summary=json.loads((hold / "summary.json").read_text()) if opened else None,
        holdout_top=pd.read_csv(hold / "top_trials.csv", index_col=0) if opened else None,
        holdout_walk_forward=(
            pd.read_csv(hold / "walk_forward.csv", index_col=0) if opened else None
        ),
    )


def filter_trials(
    trials: pd.DataFrame,
    classes: list[str] | None = None,
    families: list[str] | None = None,
    assets: list[str] | None = None,
) -> pd.DataFrame:
    """Trials matching every given filter (an empty or missing filter keeps everything)."""
    keep = pd.Series(True, index=trials.index)
    for column, wanted in (("class", classes), ("family", families), ("asset", assets)):
        if wanted:
            keep &= trials[column].isin(wanted)
    return trials.loc[keep]
