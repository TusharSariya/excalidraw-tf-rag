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
    # Needs the optional 'docling' extra.
    "test_docling_text.py::test_converter_is_cached_and_configured",
    "test_docling_text.py::test_pipeline_options_defaults",
    "test_docling_text.py::test_pipeline_options_invalid_values_fall_back",
    "test_docling_text.py::test_pipeline_options_valid_overrides",
    # Need gitignored eval data (data/eval/gold_synth/cases.json, manifest).
    "test_fold_split.py::test_both_folds_non_empty",
    "test_fold_split.py::test_determinism_identical_across_runs",
    "test_fold_split.py::test_disjoint_and_exhaustive",
    "test_fold_split.py::test_load_fold_ids_recomputes_without_file",
    "test_fold_split.py::test_per_track_balance",
    "test_fold_split.py::test_roughly_fifty_fifty",
    "test_fold_split.py::test_stratified_within_bucket",
    "test_fold_split.py::test_write_and_load_roundtrip",
    "test_gold_validation.py::test_gold_validation_references_existing_manifest_items",
    "test_gold_validation.py::test_pdf_deep_read_track_has_only_usable_relevant_labels",
    # Host-specific model tag (resolves to a GPU/mlx backend, not on a runner).
    "test_hg1_backend_match.py::test_resolved_tag_is_host_specific_for_qwen_4bit",
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
