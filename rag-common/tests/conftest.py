"""Gate integration tests behind RUN_INTEGRATION.

Tests marked ``@pytest.mark.integration`` reach the network, need live API keys,
or need a GPU / real model weights — none of which exist on a plain CI runner.
They are skipped unless ``RUN_INTEGRATION=1`` is set, so ``pytest tests/`` (and
the CI unit lane) stays offline and green. Run the full suite locally with:

    RUN_INTEGRATION=1 uv run pytest
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: needs network / live API keys / GPU (skipped unless RUN_INTEGRATION=1)",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if os.environ.get("RUN_INTEGRATION"):
        return
    skip = pytest.mark.skip(reason="integration test (set RUN_INTEGRATION=1 to run)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
