# AGENTS.md — so101-nexus

## Quick commands

```bash
make format     # ruff format + ruff check --fix
make lint       # ruff check
make typecheck  # ty check (NOT mypy/pyright)
make test       # pytest with 84% coverage threshold (COV_FAIL_UNDER=84)
```

Run `make format && make lint && make typecheck && make test` before committing. Format first; `lint` and `typecheck` can run in either order, but `test` should run last because it is slowest and depends on valid code. CI runs lint/test/docs separately and does not run `typecheck`, so the local `make typecheck` step is the safety net.

## Focused testing

```bash
uv run pytest tests/core/test_rewards.py -q                          # single file
uv run pytest tests/mujoco/test_envs.py::test_gymnasium_contract -q  # single test
uv run pytest tests/core -q                                          # single package/folder
uv run pytest -m "not slow and not visual" -q                        # skip expensive suites
```

## Test environment quirks

- Tests require headless MuJoCo rendering. `tests/conftest.py` sets `MUJOCO_GL=egl`.
- CI runs tests under `xvfb-run`. Locally, `MUJOCO_GL=egl` should suffice; if you get GL errors, prefix with `xvfb-run -a`.
- `pytest` config lives in `pyproject.toml`: `--import-mode=importlib`, `pythonpath = ["."]`, `testpaths = ["tests"]`.
- Three test markers: `slow` (integration), `visual` (LLM-verified rendering), `warp` (GPU/Warp backend).
- Visual tests need a vision model. Default is `anthropic/claude-sonnet-4-20250514`, gated by `ANTHROPIC_API_KEY`. Override with `VISUAL_TEST_MODEL` and `VISUAL_TEST_API_BASE`; supported providers also include `openai` and `openrouter`. Tests skip cleanly if no model/key is configured.
- Warp tests auto-skip if `mujoco_warp`/`torch` are absent and run on the Warp CPU device. To run: `make test-warp` or `uv run --extra warp pytest tests/warp tests/core/test_rewards_tensor.py -q`.
- Docs-consistency tests: `make docs-check` (requires `pnpm` for the Next.js docs build).

## Package layout

- **Source**: `src/so101_nexus/`
- **MuJoCo envs**: `src/so101_nexus/mujoco/` — register via `import so101_nexus.mujoco`
- **Warp envs**: `src/so101_nexus/warp/` — register via `import so101_nexus.warp`
- **Public API**: all top-level exports live in `src/so101_nexus/__init__.py`, NOT in backend submodules. Docs and user code must import from `so101_nexus`, not `so101_nexus.mujoco.*`.
- **Assets**: vendored SO101 model + MuJoCo Menagerie under `src/so101_nexus/assets/` (excluded from wheel; force-included selectively).
- **CLI**: `so101-nexus` entry point → `so101_nexus.cli:main`.

## Code style & source guardrails

- Ruff, line-length 100, target `py312`.
- NumPy-style docstrings (`[tool.ruff.lint.pydocstyle] convention = "numpy"`). Google-style `Args:` blocks are forbidden and caught by `tests/core/test_source_guardrails.py`.
- Every package `__init__.py` must define `__all__` (an empty list is acceptable).
- No em dashes, en dashes, or emoji in project-owned source, tests, or user-facing docs. `tests/core/test_source_guardrails.py` and `tests/test_docs_consistency.py` enforce this.
- Ruff enforces: `E`, `F`, `I`, `W`, `D`, `ERA`, `PLR`, `PLW`, `PLC`, `PGH004`, `C4`, `C90`, `SIM`, `RUF`, `UP`, `B`, `PIE`, `PT`, `RET`, `TCH`.
- Key ignores: `F403`/`F405` (wildcard imports in `__init__`), `PLR0913`, `PLR2004`, `PLC0415` (deferred imports in optional deps), `RUF012`, `RUF059`.
- Tests skip `D` (docstring) and `PT006` rules.

## Type checking

- Use `ty` (from the `ty` package), not pyright or mypy.
- Many third-party libs (`mujoco`, `mujoco_warp`, `cv2`, `gradio`, `plotly`, `warp`, `tyro`, `transforms3d`) are replaced with `Any` via `[tool.ty.analysis] replace-imports-with-any`.
- `pyrightconfig.json` is present for IDE support only; CI does not run it.

## Optional extras and dependency groups

| Extra / group | Purpose | Install |
|---------------|---------|---------|
| `teleop` | Leader-arm teleoperation | `uv sync --extra teleop` |
| `train` | PPO training (torch, wandb, tensorboard) | `uv sync --extra train` |
| `warp` | GPU-parallel MuJoCo Warp | `uv sync --extra warp` |
| `viz` | Pillow visualisation | `uv sync --extra viz` |
| `molmoact` | MolmoAct policy | `uv sync --extra molmoact` |
| `dev` | ruff, ty, test deps | `uv sync --group dev` |
| `visual` | litellm + imageio for visual tests | `uv sync --group visual` |

CI installs with `uv sync --locked --group dev`.

## Docs

- Next.js/Fumadocs site in `docs/`. Build with `pnpm install --frozen-lockfile && pnpm build` (CI uses pnpm 9 and Node 22).
- `make docs-check` runs the Python docs-consistency tests and then builds the Next.js site.
- `test_docs_consistency.py` guards against docs-code drift: no em dashes/emoji in user docs, env nav completeness, public API import paths, and `max_episode_steps` documented as a `gym.make` kwarg rather than a config field.

## ROCm support (AMD GPUs)

Two venvs coexist: `.venv` (CUDA, default) and `.venv-rocm` (ROCm 7.2).

```bash
make rocm-sync        # create/update .venv-rocm
make rocm-test        # run tests in ROCm venv
make rocm-test-warp   # run warp tests in ROCm venv
make rocm-format      # format in ROCm venv
make rocm-lint        # lint in ROCm venv
make rocm-typecheck   # typecheck in ROCm venv
```

Or manually: `scripts/setup-rocm.sh` (creates `.venv-rocm` from scratch). Activate: `source .venv-rocm/bin/activate`.

The setup script installs `torch` last with `--torch-backend rocm7.2 --force-reinstall` so ROCm torch is not overridden by a CUDA torch pulled in transitively.

## Gotchas

- `AGENTS.md` and `CLAUDE.md` are gitignored. Keep agent instructions local.
- Coverage threshold is 84% (`COV_FAIL_UNDER`). Run `make test` or `COV_FAIL_UNDER=0 make test` to bypass locally.
- The `warp` extra pins `mujoco-warp>=3.9.0.1,<3.10` and requires `torch`.
- Public configs/classes (`EnvironmentConfig`, `Pose`, `TouchConfig`, etc.) must be imported from `so101_nexus`, not from `so101_nexus.mujoco.*`.
- If teleop crashes during `Creating LeRobot dataset ...` with `module 'triton' has no attribute 'language'`, the `triton` wheel is incomplete. Fix with `uv pip install --force-reinstall triton`.
