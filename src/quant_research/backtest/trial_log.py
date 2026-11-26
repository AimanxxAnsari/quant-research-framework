"""A manifest for every study run: what code, what data, what settings.

Written next to the results, so any published number can be traced to the git commit that
produced it, a hash of the data it used, and the date the data runs to. Vendor data is never
copied; only its SHA-256 is recorded.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research.data.settings import Settings

PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "arch", "duckdb")


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return out.stdout.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_through(settings: Settings) -> str:
    prices = pd.read_parquet(settings.clean_dir / "prices.parquet", columns=["date"])
    return f"{prices['date'].max():%Y-%m-%d}"


def run_manifest(settings: Settings, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    clean = sorted(settings.clean_dir.glob("*.parquet"))
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git(settings.root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(settings.root, "status", "--porcelain", "--", "src", "config")),
        "data_through": data_through(settings),
        "data_sha256": {p.name: _sha256(p) for p in clean},
        "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in PACKAGES},
        **(extra or {}),
    }
