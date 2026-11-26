.PHONY: install hooks test lint typecheck fetch build check refresh app

install:
	uv sync --all-groups

hooks:
	uv run pre-commit install

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy

fetch:
	uv run qrf fetch

build:
	uv run qrf build

check:
	uv run qrf check

refresh:
	uv run qrf refresh

app:
	uv run qrf app
