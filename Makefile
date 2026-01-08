.PHONY: check format

check:
	uv run ruff check --fix
	uv run ty check

format:
	uv run ruff check --select I --fix
	uv run ruff format