.PHONY: install test lint fetch build check refresh

install:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

fetch:
	uv run qrf fetch

build:
	uv run qrf build

check:
	uv run qrf check

refresh:
	uv run qrf refresh
