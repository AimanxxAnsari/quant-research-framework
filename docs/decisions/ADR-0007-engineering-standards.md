# ADR-0007: Engineering standards before the backtester

- **Date:** 2026-11-26
- **Status:** accepted

## Context
Pillar 4 adds a backtesting engine whose output (Sharpe ratios, deflated Sharpe, PBO) is only as
trustworthy as the code under it. Until now the repo had lint, formatting and tests, but errors
were plain `ValueError` / `KeyError`, nothing checked types, CI ran on one Python version, and
nothing stopped a private file, an API key or a junk file from being committed by mistake.

## Decisions

| Topic | Choice | Rejected alternative and why |
|---|---|---|
| Errors | One small family in `quant_research.errors`: `ResearchInputError` (a `ValueError`) with `EmptySeriesError`, `MisalignedDataError`, `MissingValuesError`, `InvalidParameterError`, `ConfigError`; plus `DataNotFoundError` (a `KeyError`) and `DataIntegrityError` | One class per failure: too many to remember. Bare `ValueError`: callers cannot tell a bad parameter from corrupt data |
| Compatibility | Each error also subclasses the built-in it replaces, and existing raises were switched over | A clean break: would silently change what existing `except ValueError` code catches |
| Type checking | mypy, strict on core maths (`stats`, `risk`, `validation`, `backtest`), standard mode elsewhere | Strict everywhere: pandas stubs make plotting and study scripts noisy for little gain. pyright: fine too, but mypy already fits the uv toolchain |
| numpy on pandas | Typed helpers `log` and `sqrt` in `quant_research._typing` | `cast` in every formula: noisy and easy to get wrong |
| Strict flags | Listed one by one in the mypy override | `strict = true` inside an override: in mypy 2.4 it leaked to every module (found while setting this up) |
| Pre-commit | pre-commit-hooks (whitespace, YAML/TOML, merge markers, large files, private keys) plus local hooks that run ruff and mypy from `uv.lock` and a hygiene script | Upstream ruff/mypy hook repos: a second place to pin versions that can drift from the lockfile |
| Hygiene | `scripts/check_hygiene.py` fails on private folders (`1-brain`, `local-brain`), `.env` and `.DS_Store`, any `.env` key value found in a file, and em dashes | Relying on `.gitignore` alone: it does not stop `git add -f` or a key pasted into code |
| CI | Python 3.12 and 3.13 matrix: hygiene, ruff, format, mypy, tests | Single version: 3.13 differences would only show up later |

## Consequences
- Rebuilding the Pillar 2 and Pillar 5 studies after the refactor gave a byte-identical README and
  identical figures, so no published number changed.
- New maths modules must pass strict typing from their first commit.
- The hygiene check only finds leaked key values on a machine that has a `.env`; in CI it still
  checks paths and em dashes.
