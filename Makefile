SHELL = bash
ifneq ("$(wildcard .env)","")
	include .env
endif

.PHONY: setup build-index run benchmark lint format type-check

.ONESHELL:
setup:
	uv sync
	uv run python scripts/build_rag_index.py

.ONESHELL:
build-index:
	uv run python scripts/build_rag_index.py

.ONESHELL:
run:
	uv run uvicorn src.server:app --host 0.0.0.0 --port 8000

.ONESHELL:
benchmark:
	uv run python scripts/benchmark_dev.py

.ONESHELL:
lint:
	uv run ruff check --fix --exit-non-zero-on-fix --config=ruff.toml

.ONESHELL:
format:
	uv run ruff format --check --config=ruff.toml

.ONESHELL:
type-check:
	uv run pyright
