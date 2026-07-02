"""Smoke test for the `map` CLI group wiring (the litmap logic itself is unit-tested in
rag-common/tests/test_litmap_*.py)."""

from __future__ import annotations

from click.testing import CliRunner
from rag_literature_rag.cli import main
from rag_literature_rag.query.litmap_cli import map_group


def test_map_group_registered_with_expected_subcommands():
    assert "map" in main.commands
    assert set(map_group.commands) == {"clusters", "lineage", "similar", "seminal"}


def test_map_help_runs():
    result = CliRunner().invoke(main, ["map", "--help"])
    assert result.exit_code == 0
    for sub in ("clusters", "lineage", "similar", "seminal"):
        assert sub in result.output
