.PHONY: format lint typecheck test test-warp test-visual test-visual-qwen coverage docs-check
.PHONY: rocm-sync rocm-test rocm-test-warp rocm-format rocm-lint rocm-typecheck

COV_FAIL_UNDER ?= 84
export COV_FAIL_UNDER

# ---------------------------------------------------------------------------
# CUDA (default) — uses .venv
# ---------------------------------------------------------------------------

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff check

typecheck:
	uv run ty check

test:
	uv run pytest --cov=so101_nexus --cov-report=term-missing --cov-fail-under=$(COV_FAIL_UNDER)

test-warp:
	uv run --extra warp pytest tests/warp tests/core/test_rewards_tensor.py -q

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

# ---------------------------------------------------------------------------
# ROCm — uses .venv-rocm (created by scripts/setup-rocm.sh)
#
#   make rocm-sync          # create/update .venv-rocm with ROCm torch
#   make rocm-test          # run tests in the ROCm venv
#   make rocm-test-warp     # run warp tests in the ROCm venv
#   make rocm-format        # format in the ROCm venv
#   make rocm-lint          # lint in the ROCm venv
#   make rocm-typecheck     # typecheck in the ROCm venv
# ---------------------------------------------------------------------------

ROCM_PY := .venv-rocm/bin/python

rocm-sync:
	scripts/setup-rocm.sh

rocm-test:
	$(ROCM_PY) -m pytest --cov=so101_nexus --cov-report=term-missing --cov-fail-under=$(COV_FAIL_UNDER)

rocm-test-warp:
	$(ROCM_PY) -m pytest tests/warp tests/core/test_rewards_tensor.py -q

rocm-format:
	.venv-rocm/bin/ruff format
	.venv-rocm/bin/ruff check --fix

rocm-lint:
	.venv-rocm/bin/ruff check

rocm-typecheck:
	.venv-rocm/bin/ty check
