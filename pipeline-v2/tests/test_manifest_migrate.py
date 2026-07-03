"""Manifest migration tests.

Pure-logic pieces (canonicalization, diffing) are plain unit tests — no DB
needed. The actual migrate-into-postgres / export-back-out round trip needs
a real database and is an integration test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pipeline_v2 import manifest_migrate

FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v1_sample.json"


def test_load_v1_manifest_json_reads_real_field_shape():
    manifest = manifest_migrate.load_v1_manifest_json(FIXTURE)
    assert manifest["version"] == 1
    assert len(manifest["items"]) == 3
    ids = {item["id"] for item in manifest["items"]}
    assert ids == {"smith2020-graph-drawing", "doe2019-sugiyama", "graphviz-impl"}


def test_canonical_json_is_order_independent():
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert manifest_migrate.canonical_json(a) == manifest_migrate.canonical_json(b)


def test_diff_manifests_no_drift_when_items_match():
    manifest = manifest_migrate.load_v1_manifest_json(FIXTURE)
    # export_from_db always resets updatedAt; diff_manifests must ignore it.
    exported = {"version": 1, "updatedAt": "", "items": list(manifest["items"])}

    report = manifest_migrate.diff_manifests(manifest, exported)

    assert report.drift is False
    assert report.input_item_count == 3
    assert report.exported_item_count == 3
    assert report.missing_ids == []
    assert report.extra_ids == []
    assert report.changed_ids == []


def test_diff_manifests_detects_missing_item():
    manifest = manifest_migrate.load_v1_manifest_json(FIXTURE)
    exported_items = [i for i in manifest["items"] if i["id"] != "doe2019-sugiyama"]
    exported = {"version": 1, "updatedAt": "", "items": exported_items}

    report = manifest_migrate.diff_manifests(manifest, exported)

    assert report.drift is True
    assert report.missing_ids == ["doe2019-sugiyama"]
    assert report.extra_ids == []
    assert report.changed_ids == []


def test_diff_manifests_detects_changed_payload():
    manifest = manifest_migrate.load_v1_manifest_json(FIXTURE)
    items = [dict(i) for i in manifest["items"]]
    items[0]["title"] = "A mutated title"
    exported = {"version": 1, "updatedAt": "", "items": items}

    report = manifest_migrate.diff_manifests(manifest, exported)

    assert report.drift is True
    assert report.changed_ids == ["smith2020-graph-drawing"]


def test_diff_manifests_detects_extra_item():
    manifest = manifest_migrate.load_v1_manifest_json(FIXTURE)
    items = list(manifest["items"]) + [
        {"id": "unexpected-extra", "title": "x", "source": "s", "url": "u", "status": "ok"}
    ]
    exported = {"version": 1, "updatedAt": "", "items": items}

    report = manifest_migrate.diff_manifests(manifest, exported)

    assert report.drift is True
    assert report.extra_ids == ["unexpected-extra"]


def test_export_to_json_file_round_trips_through_disk(tmp_path):
    manifest = manifest_migrate.load_v1_manifest_json(FIXTURE)
    out_path = tmp_path / "exported.json"

    manifest_migrate.export_to_json_file(manifest, out_path)
    reloaded = manifest_migrate.load_v1_manifest_json(out_path)

    report = manifest_migrate.diff_manifests(manifest, reloaded)
    assert report.drift is False


@pytest.mark.integration
def test_migrate_then_export_round_trips_byte_identical(clean_postgres_dsn):
    from pipeline_v2.workstate import PostgresWorkStateRepository

    PostgresWorkStateRepository(clean_postgres_dsn).init_db()

    manifest = manifest_migrate.load_v1_manifest_json(FIXTURE)
    count = manifest_migrate.migrate_to_db(manifest, corpus="glr-fixture", dsn=clean_postgres_dsn)
    assert count == 3

    exported = manifest_migrate.export_from_db(corpus="glr-fixture", dsn=clean_postgres_dsn)
    assert len(exported["items"]) == 3

    report = manifest_migrate.diff_manifests(manifest, exported)
    assert report.drift is False, report.summary()
