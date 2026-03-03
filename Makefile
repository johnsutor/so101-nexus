.PHONY: format lint typecheck test test-mujoco test-maniskill test-visual test-visual-mujoco test-visual-maniskill test-visual-qwen

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

test-visual: test-visual-mujoco test-visual-maniskill

test-visual-mujoco:
	uv run --package so101-nexus-mujoco --with litellm --with Pillow --with "imageio[ffmpeg]" pytest tests/visual/test_mujoco_visual.py -m visual -v

test-visual-maniskill:
	uv run --package so101-nexus-maniskill --prerelease=allow --with litellm --with Pillow --with "imageio[ffmpeg]" pytest tests/visual/test_maniskill_visual.py -m visual -v

test-visual-qwen:
	VISUAL_TEST_MODEL="openai/unsloth/Qwen3_5-35B-A3B-UD-Q4_K_M" VISUAL_TEST_API_BASE="http://127.0.0.1:1337/v1" OPENAI_API_KEY="secret" $(MAKE) test-visual