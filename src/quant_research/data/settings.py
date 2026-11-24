"""Paths, configuration and secrets.

Secrets come from environment variables, loaded from a git-ignored ``.env`` file at the
repository root. They are never logged or written to disk by the framework.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml
from dotenv import load_dotenv


def repo_root(start: Path | None = None) -> Path:
    """Nearest parent directory containing pyproject.toml and config/."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() and (p / "config").is_dir():
            return p
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    config_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.data_dir / "clean"

    @property
    def warehouse(self) -> Path:
        return self.data_dir / "warehouse.duckdb"

    def instruments(self) -> dict:
        return yaml.safe_load((self.config_dir / "instruments.yaml").read_text())

    def sources(self) -> dict:
        return yaml.safe_load((self.config_dir / "sources.yaml").read_text())["sources"]


def load_settings(root: Path | None = None, data_dir: Path | None = None) -> Settings:
    root = root or repo_root()
    load_dotenv(root / ".env", override=False)
    data = data_dir or Path(os.environ.get("QRF_DATA_DIR", root / "data"))
    return Settings(root=root, data_dir=Path(data), config_dir=root / "config")


@cache
def secret(name: str) -> str:
    """Read an API key from the environment; fail clearly (without echoing values)."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to .env (see .env.example).")
    return value
