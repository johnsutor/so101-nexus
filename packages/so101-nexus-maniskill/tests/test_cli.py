"""Tests for the maniskill-backend argparse CLI."""

from __future__ import annotations

import pytest

from so101_nexus_core.testing.cli_contract import (
    run_main_dispatch_contract,
    run_parser_contract,
    run_parser_requires_subcommand,
)
from so101_nexus_maniskill import cli as maniskill_cli


def test_build_parser_has_teleop_subcommand():
    run_parser_contract(maniskill_cli)


def test_build_parser_requires_subcommand():
    run_parser_requires_subcommand(maniskill_cli, pytest=pytest)


def test_main_dispatches_teleop(monkeypatch):
    run_main_dispatch_contract(
        maniskill_cli,
        backend="maniskill",
        argv0="so101-nexus-maniskill",
        monkeypatch=monkeypatch,
    )
