.PHONY: sync
sync:
	uv sync

.PHONY: format
format:
	uv run ruff format .

.PHONY: lint
lint:
	uv run ruff check . --fix

.PHONY: all
all: sync format lint

.PHONY: run
run:
	uv run main.py

.PHONY: simple-agent
run-simple-agent:
	uv run examples/langgraph/simple_agent.py

.PHONY: mcps
run-mcps:
	uv run examples/fastmcp/mcps.py