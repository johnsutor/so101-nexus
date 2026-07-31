#!/usr/bin/env python3
"""Publish the ``envhub/`` package to a Hugging Face Hub model repository.

The uploaded files are the LeRobot EnvHub entry points for the SO101-Nexus
environments; the environment code itself ships in the ``so101-nexus`` package
that ``envhub/requirements.txt`` pins.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import tyro

ENVHUB_DIR = Path(__file__).resolve().parents[1] / "envhub"
IGNORE_PATTERNS = ["__pycache__/*", "*.pyc"]


@dataclass
class Args:
    """Command-line arguments for the EnvHub publisher."""

    repo_id: str = "johnsutor/so101-nexus-envs"
    private: bool = False
    commit_message: str = "Publish SO101-Nexus EnvHub entry points"
    dry_run: bool = False


def _check_requirement_pin() -> None:
    """Warn when the pinned minimum version is not released yet."""
    from packaging.version import Version

    from so101_nexus import __version__

    pin = re.search(r"^so101-nexus>=(\S+)$", (ENVHUB_DIR / "requirements.txt").read_text(), re.M)
    if pin and Version(__version__.split("+")[0]) < Version(pin.group(1)):
        print(
            f"warning: requirements.txt pins so101-nexus>={pin.group(1)} but the installed "
            f"version is {__version__}. Publish after that release."
        )


def main(args: Args) -> None:
    """Create the repository if needed and upload ``envhub/``."""
    files = sorted(
        path.relative_to(ENVHUB_DIR).as_posix()
        for path in ENVHUB_DIR.rglob("*.*")
        if "__pycache__" not in path.parts
    )
    print(f"uploading {len(files)} files from {ENVHUB_DIR} to {args.repo_id}:")
    for name in files:
        print(f"  {name}")
    _check_requirement_pin()
    if args.dry_run:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(ENVHUB_DIR),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"published https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
