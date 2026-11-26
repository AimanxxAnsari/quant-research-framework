"""Repository hygiene check, run by pre-commit on staged files and by CI on every tracked file.

It fails (exit code 1) when a file:

* sits in a private folder (``1-brain/``, ``local-brain/``) or is a secrets or junk file
  (``.env``, ``.DS_Store``),
* contains the value of an API key from the local ``.env`` (checked only where ``.env`` exists,
  so on a developer machine; key values are never printed),
* contains an em dash (house style: none anywhere in the repo).

Usage: ``python scripts/check_hygiene.py FILE [FILE ...]``. With no arguments it checks every
file tracked by git.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path, PurePosixPath

PRIVATE_PARTS = {"1-brain", "local-brain"}
FORBIDDEN_NAMES = {".env", ".DS_Store"}
ALLOWED_ENV = {".env.example"}
EM_DASH = "\u2014"
MIN_SECRET_LENGTH = 16  # shorter values are too likely to match ordinary text
SKIP_CONTENT = {"uv.lock"}  # machine-written; third-party metadata is not ours to restyle


def _secret_values(root: Path) -> list[str]:
    env = root / ".env"
    if not env.exists():
        return []
    values = []
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        value = line.split("=", 1)[1].strip().strip("'\"")
        if len(value) >= MIN_SECRET_LENGTH:
            values.append(value)
    return values


def _path_problem(path: PurePosixPath) -> str | None:
    if PRIVATE_PARTS & set(path.parts):
        return "file is inside a private folder"
    if path.name in ALLOWED_ENV:
        return None
    if path.name in FORBIDDEN_NAMES or path.name.startswith(".env."):
        return "secrets or junk file must not be committed"
    return None


def _content_problems(path: Path, secrets: list[str]) -> list[str]:
    if path.name in SKIP_CONTENT or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []  # binary file (a chart, for example)
    problems = []
    if any(s in text for s in secrets):
        problems.append("contains an API key value from .env")
    if EM_DASH in text:
        line = next(i for i, ln in enumerate(text.splitlines(), 1) if EM_DASH in ln)
        problems.append(f"contains an em dash (first on line {line})")
    return problems


def check(files: list[str], root: Path) -> list[str]:
    secrets = _secret_values(root)
    failures = []
    for name in files:
        rel = PurePosixPath(name)
        problem = _path_problem(rel)
        if problem:
            failures.append(f"{name}: {problem}")
            continue
        failures += [f"{name}: {p}" for p in _content_problems(root / name, secrets)]
    return failures


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    files = (
        argv
        or subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    failures = check(files, root)
    for f in failures:
        print(f"hygiene: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
