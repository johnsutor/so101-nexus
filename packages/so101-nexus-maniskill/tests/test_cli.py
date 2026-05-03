"""Tests for the maniskill-backend argparse CLI."""

from __future__ import annotations

from so101_nexus_core.testing.cli_contract import (
    run_main_dispatches_teleop,
    run_parser_contract,
    run_parser_requires_subcommand,
    run_parser_wrist_roll_offset_contract,
)
from so101_nexus_maniskill import cli as maniskill_cli


def test_build_parser_has_teleop_subcommand():
    run_parser_contract(maniskill_cli)


def test_build_parser_wrist_roll_offset_parses():
    run_parser_wrist_roll_offset_contract(maniskill_cli)


def test_build_parser_requires_subcommand():
    run_parser_requires_subcommand(maniskill_cli)


def test_main_dispatches_teleop(monkeypatch):
    run_main_dispatches_teleop(
        maniskill_cli,
        backend="maniskill",
        monkeypatch=monkeypatch,
        argv_prog="so101-nexus-maniskill",
    )
