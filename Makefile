.PHONY: sync
sync:
	uv sync

.PHONY: bootstrap
pre-commit:
	uv run pre-commit install

.PHONY: format
format:
	uv run ruff format .

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: all
all: sync pre-commit format lint

.PHONY: simple-agent
run-simple-agent:
	uv run examples/langgraph/simple_agent.py

.PHONY: mcps
run-mcps:
	uv run examples/fastmcp/mcps.py