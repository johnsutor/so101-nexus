# Contributing to SO101-Nexus

Thanks for your interest in improving SO101-Nexus. This guide covers the
development setup, the checks your change must pass, and the release process.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) and targets Python 3.12+.

```bash
git clone https://github.com/johnsutor/so101-nexus.git
cd so101-nexus
uv sync                 # add --extra teleop / --extra train / --extra warp as needed
```

## Checks

Run these before opening a pull request:

```bash
make format             # ruff format and autofix
make lint               # ruff lint
make typecheck          # ty
make test               # full suite with the coverage gate
```

Run the narrowest relevant test set first, then broaden:

```bash
uv run pytest tests/mujoco/test_envs.py -q
make test-warp          # Warp backend tests (needs the warp extra)
```

## Testing policy

- Bug fix: add a regression test that fails before the fix and passes after.
- New behavior: cover the happy path plus at least one edge case.
- Refactor: no behavior change; existing tests stay green.
- Never weaken assertions, types, or validation to make a test pass.

## Style and conventions

- Follow the existing patterns in the codebase; if none exists, propose one first.
- Comments explain "why", not "what". Public modules, classes, and functions get
  concise NumPy-style docstrings.
- Degrees, not radians, in all public, config, and user-facing APIs. Internally use
  explicit `_deg` / `_rad` suffixes and never mix silently.
- New core abstractions (rewards, observation shaping, transforms) must accept both
  NumPy arrays and torch tensors so the Warp backend can use them.
- Seeded RNG only for reset-time randomness: `self.np_random` (MuJoCo) or a seeded
  torch RNG. Never global `np.random` or Python `random`.
- Reward logic lives in `so101_nexus/rewards.py`; never inline reward formulas in
  backend env classes.
- Prefer LeRobot abstractions over bespoke equivalents. Custom processor steps
  subclass the LeRobot typed bases and register with `ProcessorStepRegistry`.

See the [Stability and versioning](https://so101-nexus.com/docs/getting-started/stability)
page for the public-API policy.

## Branches and commits

Branch off `main` with a `fix/`, `feat/`, `refactor/`, or `chore/` prefix. Use
[Conventional Commits](https://www.conventionalcommits.org/) messages and commit each
meaningful step. Use standard git commits; never commit via credential tokens.

## Pull requests

- Keep pull requests focused and describe what changed and why.
- Update tests and documentation alongside code.
- Add a `## [Unreleased]` entry to [CHANGELOG.md](CHANGELOG.md) for any user-facing
  change.
- Continuous integration runs lint, tests (Python 3.12 and 3.13, Linux and macOS),
  the Warp backend tests, and the documentation build. All must pass.

## Release process

Releases are published to PyPI by the `Publish to PyPI` workflow when a GitHub
release is published.

1. Bump `version` in `pyproject.toml`.
2. Move the `## [Unreleased]` changelog entries under a new dated version heading and
   update the comparison links at the bottom of [CHANGELOG.md](CHANGELOG.md).
3. Open a release-preparation pull request and merge it once green.
4. Tag the release and publish it on GitHub. The publish workflow builds and uploads
   the distributions via trusted publishing, and the release smoke test installs the
   published package and runs a backend rollout.
