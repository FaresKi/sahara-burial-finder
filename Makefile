.PHONY: all test lint plan

# Reproducible, offline, public-only rebuild from data/raw (SPEC 9.3). Restricted stages stay off.
all:
	uv run sbf run-all

plan:
	uv run sbf run-all --dry-run

test:
	uv run pytest

lint:
	uv run ruff check src tests
