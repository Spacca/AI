.PHONY: run
run:
	poetry run python simple_agent.py

.PHONY: install
install:
	poetry install --no-root
