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
    # Needs a live Ollama server.
    "test_local_llm.py::test_generate_text_ollama_parses_openai_response",
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
