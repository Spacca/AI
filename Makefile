.PHONY: run
run:
	poetry run python human_input.py

.PHONY: install
install:
	poetry install --no-root
