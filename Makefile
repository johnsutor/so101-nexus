.PHONY: format lint typecheck test test-visual test-visual-qwen coverage docs-check

COV_FAIL_UNDER ?= 84
export COV_FAIL_UNDER

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff check

typecheck:
	uv run ty check

test:
	uv run pytest --cov=so101_nexus --cov-report=term-missing --cov-fail-under=$(COV_FAIL_UNDER)

coverage:
	uv run pytest --cov=so101_nexus --cov-report=term-missing --cov-report=html --cov-fail-under=$(COV_FAIL_UNDER)

test-visual:
	uv run --with litellm --with Pillow --with "imageio[ffmpeg]" \
	  pytest tests/visual/test_mujoco_visual.py -m visual -v

test-visual-qwen:
	VISUAL_TEST_MODEL="openai/unsloth/Qwen3_5-35B-A3B-UD-Q4_K_M" \
	VISUAL_TEST_API_BASE="http://127.0.0.1:1337/v1" OPENAI_API_KEY="secret" $(MAKE) test-visual

docs-check:
	uv run pytest tests/test_docs_consistency.py
	cd docs && pnpm install --frozen-lockfile && pnpm build
