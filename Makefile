.PHONY: format lint typecheck

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff check

typecheck:
	uv run ty check

test:
	uv run pytest --cov=so101_nexus tests/