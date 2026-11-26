"""The repository hygiene check (scripts/check_hygiene.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene", ROOT / "scripts/check_hygiene.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hygiene_flags_private_secret_and_junk_paths(tmp_path):
    h = _hygiene()
    files = ["local-brain/notes.md", "docs/1-brain/x.md", ".env", ".env.local", ".DS_Store"]
    failures = h.check(files, tmp_path)
    assert len(failures) == len(files)
    assert h.check([".env.example"], tmp_path) == []


def test_hygiene_flags_em_dashes_and_leaked_keys(tmp_path):
    h = _hygiene()
    key = "abcdef0123456789abcdef"
    (tmp_path / ".env").write_text(f"FRED_API_KEY={key}\nSHORT=abc\n")
    (tmp_path / "clean.md").write_text("plain text, short value abc\n")
    (tmp_path / "dash.md").write_text("one\ntwo \u2014 three\n")
    (tmp_path / "leak.py").write_text(f'KEY = "{key}"\n')
    failures = h.check(["clean.md", "dash.md", "leak.py"], tmp_path)
    assert failures == [
        "dash.md: contains an em dash (first on line 2)",
        "leak.py: contains an API key value from .env",
    ]
    assert all(key not in f for f in failures)  # the key itself is never echoed
