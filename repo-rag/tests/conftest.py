"""Gate integration tests behind RUN_INTEGRATION.

A test is "integration" if it can't run on a bare CI runner — it reaches the
network, needs a live API key / Ollama / GPU, needs an optional extra, or needs
gitignored data. Those are auto-marked below (by node id, so the test files stay
untouched) and skipped unless ``RUN_INTEGRATION=1`` is set, keeping the default
run and the CI unit lane offline and green:

    RUN_INTEGRATION=1 uv run pytest     # everything, incl. live/network/GPU

The set is seeded from a CI run with sockets disabled (`--disable-socket`); add
to it if you add a test that needs the network / an API / the GPU / real data.
"""

from __future__ import annotations

import os

import pytest

# "test_file.py::test_name" (node id minus the tests/ dir).
_INTEGRATION = {
    # Network: tiktoken downloads its BPE vocab; graph build resolves online.
    "test_chunk.py::test_count_tokens_positive",
    "test_chunk.py::test_typescript_chunks_at_function_boundaries",
    "test_graph.py::test_graph_finds_importers_and_callers",
    # Needs the real excalidraw repo checkout to scan.
    "test_harvest.py::test_classify_terraform_source_type",
    "test_harvest.py::test_harvest_includes_handoff_and_terraform",
    "test_harvest.py::test_harvest_keeps_only_english_locale",
    # Needs the optional 'watch' extra (watchfiles).
    "test_watch.py::test_watch_requires_pinned_profile",
}


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: needs network / live API keys / GPU / extras / real data "
        "(skipped unless RUN_INTEGRATION=1)",
    )


def pytest_collection_modifyitems(config, items) -> None:
    for item in items:
        if item.nodeid.split("/")[-1] in _INTEGRATION:
            item.add_marker(pytest.mark.integration)
    if os.environ.get("RUN_INTEGRATION"):
        return
    skip = pytest.mark.skip(reason="integration test (set RUN_INTEGRATION=1 to run)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
