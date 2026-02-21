.PHONY: format lint typecheck test test-mujoco test-maniskill

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff check

typecheck:
	uv run ty check

test: test-mujoco test-maniskill

test-mujoco:
	uv run --package so101-nexus-mujoco pytest packages/so101-nexus-mujoco/tests/ --cov

test-maniskill:
	uv run --package so101-nexus-maniskill --prerelease=allow pytest packages/so101-nexus-maniskill/tests/ --cov
