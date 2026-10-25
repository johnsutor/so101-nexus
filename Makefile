.PHONY: format lint typecheck test test-warp test-visual test-visual-qwen coverage docs-check rocm-sync rocm-verify

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

test-warp:
	uv run --extra warp pytest tests/warp tests/core/test_rewards_tensor.py -q

rocm-sync:
	uv sync --extra rocm --no-default-groups

rocm-verify: rocm-sync
	uv run --extra rocm --no-default-groups python -c "\
import torch; \
hip = getattr(torch.version, 'hip', None); \
assert hip, 'ROCm/HIP not detected in this torch build'; \
print(f'torch {torch.__version__}  ROCm/HIP {hip}  GPU available: {torch.cuda.is_available()}')"

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
