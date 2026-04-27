.PHONY: format lint typecheck test test-core test-mujoco test-maniskill \
        test-visual test-visual-mujoco test-visual-maniskill test-visual-qwen \
        coverage clean-coverage

COV_FAIL_UNDER ?= 84
export COV_FAIL_UNDER

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff check

typecheck:
	uv run ty check

test: clean-coverage
	$(MAKE) -j3 test-core test-mujoco test-maniskill
	uv run coverage combine
	uv run coverage report --fail-under=$(COV_FAIL_UNDER)

clean-coverage:
	rm -rf .coverage .coverage.* htmlcov

test-core:
	COVERAGE_FILE=.coverage.core uv run --package so101-nexus-core --prerelease=allow pytest packages/so101-nexus-core/tests \
	  --cov=so101_nexus_core --cov-report=

test-mujoco:
	COVERAGE_FILE=.coverage.mujoco uv run --package so101-nexus-mujoco --prerelease=allow pytest packages/so101-nexus-mujoco/tests \
	  --cov=so101_nexus_mujoco --cov=so101_nexus_core --cov-report=

test-maniskill:
	COVERAGE_FILE=.coverage.maniskill uv run --package so101-nexus-maniskill --prerelease=allow pytest \
	  packages/so101-nexus-maniskill/tests --cov=so101_nexus_maniskill --cov=so101_nexus_core --cov-report=

coverage:
	uv run coverage combine
	uv run coverage report --fail-under=$(COV_FAIL_UNDER)
	uv run coverage html

test-visual: test-visual-mujoco test-visual-maniskill

test-visual-mujoco:
	uv run --package so101-nexus-mujoco --with litellm --with Pillow --with "imageio[ffmpeg]" \
	  pytest tests/visual/test_mujoco_visual.py -m visual -v

test-visual-maniskill:
	uv run --package so101-nexus-maniskill --prerelease=allow --with litellm --with Pillow \
	  --with "imageio[ffmpeg]" pytest tests/visual/test_maniskill_visual.py -m visual -v

test-visual-qwen:
	VISUAL_TEST_MODEL="openai/unsloth/Qwen3_5-35B-A3B-UD-Q4_K_M" \
	VISUAL_TEST_API_BASE="http://127.0.0.1:1337/v1" OPENAI_API_KEY="secret" $(MAKE) test-visual
