"""Tests for `pipeline-v2 enqueue` (T0c).

Offline tests exercise the manifest/glob item-extraction helpers directly
and drive the click command via ``CliRunner`` with ``build_queue`` swapped
for ``FakeWorkQueue`` (no Kafka needed). The integration test is a real
round trip: invoke the CLI against a live Kafka broker, then consume
``docs.enriched`` (a per-test-suffixed topic, registered into
``contracts.TOPIC_MODELS`` the same way ``test_messaging_integration.py``'s
``kafka_topics`` fixture does) and assert every produced message is a
well-formed ``DocsEnrichedMessage``.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner
from fake_queue import FakeWorkQueue
from pipeline_v2 import cli as cli_module
from pipeline_v2 import contracts as contracts_module
from pipeline_v2.cli import _iter_glob_enqueue_items, _iter_manifest_enqueue_items, main
from pipeline_v2.contracts import DocsEnrichedMessage
from pipeline_v2.manifest_migrate import load_v1_manifest_json

FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v1_sample.json"


# --------------------------------------------------------------------------
# Item-extraction helpers.
# --------------------------------------------------------------------------


def test_iter_manifest_enqueue_items_skips_items_without_local_path():
    manifest = load_v1_manifest_json(FIXTURE)

    items = _iter_manifest_enqueue_items(manifest)

    # Only "smith2020-graph-drawing" has a localPath; the metadata_only and
    # failed items (null localPath) are not enqueueable.
    assert [item["doc_id"] for item in items] == ["smith2020-graph-drawing"]


def test_iter_manifest_enqueue_items_uses_existing_sha256_without_hashing_file():
    manifest = load_v1_manifest_json(FIXTURE)

    items = _iter_manifest_enqueue_items(manifest)

    assert items[0]["sha256"] == "a" * 64
    assert items[0]["pdf_path"] == "papers/smith2020.pdf"
    assert items[0]["title"] == "A Survey of Graph Drawing Algorithms"
    assert items[0]["source_url"] == "https://example.org/smith2020"


def test_iter_manifest_enqueue_items_hashes_file_when_sha256_missing(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf bytes")
    manifest = {
        "version": 1,
        "items": [{"id": "d1", "localPath": str(pdf_path), "sha256": None}],
    }

    items = _iter_manifest_enqueue_items(manifest)

    import hashlib

    assert items[0]["sha256"] == hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def test_iter_glob_enqueue_items_hashes_real_files(tmp_path):
    (tmp_path / "b.pdf").write_bytes(b"bbb")
    (tmp_path / "a.pdf").write_bytes(b"aaa")
    (tmp_path / "not-a-pdf.txt").write_bytes(b"ignored")

    items = _iter_glob_enqueue_items(str(tmp_path / "*.pdf"))

    assert [item["doc_id"] for item in items] == ["a", "b"]  # sorted by path
    import hashlib

    assert items[0]["sha256"] == hashlib.sha256(b"aaa").hexdigest()


# --------------------------------------------------------------------------
# CLI command, offline (FakeWorkQueue swapped in for build_queue).
# --------------------------------------------------------------------------


@pytest.fixture
def fake_build_queue(monkeypatch):
    fake = FakeWorkQueue()

    def _build_queue(backend, *, bootstrap_servers, group_id, topics, **kwargs):
        return fake

    monkeypatch.setattr(cli_module, "build_queue", _build_queue, raising=False)
    # enqueue_cmd imports build_queue locally from pipeline_v2.messaging; patch there too.
    import pipeline_v2.messaging as messaging_module

    monkeypatch.setattr(messaging_module, "build_queue", _build_queue)
    return fake


def test_enqueue_cmd_requires_exactly_one_source():
    runner = CliRunner()
    result = runner.invoke(main, ["enqueue", "--corpus", "glr"])
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_enqueue_cmd_rejects_both_sources(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "enqueue",
            "--corpus",
            "glr",
            "--manifest",
            str(FIXTURE),
            "--pdf-glob",
            str(tmp_path / "*.pdf"),
        ],
    )
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_enqueue_cmd_produces_well_formed_messages_from_manifest(fake_build_queue):
    runner = CliRunner()
    result = runner.invoke(
        main, ["enqueue", "--corpus", "glr", "--manifest", str(FIXTURE)]
    )

    assert result.exit_code == 0, result.output
    assert len(fake_build_queue.produced) == 1
    topic, value = fake_build_queue.produced[0]
    assert topic == "docs.enriched"
    message = DocsEnrichedMessage.model_validate_json(value)
    assert message.doc_id == "smith2020-graph-drawing"
    assert message.corpus == "glr"
    assert message.payload["pdf_path"] == "papers/smith2020.pdf"
    assert fake_build_queue.closed


def test_enqueue_cmd_respects_limit(fake_build_queue, tmp_path):
    for i in range(5):
        (tmp_path / f"doc{i}.pdf").write_bytes(f"content-{i}".encode())

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "enqueue",
            "--corpus",
            "glr",
            "--pdf-glob",
            str(tmp_path / "*.pdf"),
            "--limit",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(fake_build_queue.produced) == 2


def test_enqueue_cmd_no_items_produces_nothing(fake_build_queue, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["enqueue", "--corpus", "glr", "--pdf-glob", str(tmp_path / "*.pdf")],
    )

    assert result.exit_code == 0, result.output
    assert fake_build_queue.produced == []
    assert "no enqueueable items" in result.output


def test_enqueue_cmd_stamps_custom_stage_fingerprint(fake_build_queue):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "enqueue",
            "--corpus",
            "glr",
            "--manifest",
            str(FIXTURE),
            "--stage-fingerprint",
            "custom-fp",
        ],
    )

    assert result.exit_code == 0, result.output
    _topic, value = fake_build_queue.produced[0]
    assert json.loads(value)["stage_fingerprint"] == "custom-fp"


# --------------------------------------------------------------------------
# Integration: real round trip through a live Kafka broker.
# --------------------------------------------------------------------------


@pytest.fixture
def docs_enriched_topic(kafka_bootstrap: str, monkeypatch: pytest.MonkeyPatch) -> str:
    from confluent_kafka.admin import AdminClient, NewTopic

    name = f"docs.enriched.{uuid.uuid4().hex[:8]}"
    admin = AdminClient({"bootstrap.servers": kafka_bootstrap})
    futures = admin.create_topics([NewTopic(name, num_partitions=1, replication_factor=1)])
    for _name, future in futures.items():
        try:
            future.result(timeout=20)
        except Exception as exc:  # noqa: BLE001
            if "already exists" not in str(exc).lower():
                raise
    monkeypatch.setitem(contracts_module.TOPIC_MODELS, name, DocsEnrichedMessage)
    return name


@pytest.mark.integration
def test_enqueue_cli_round_trips_n_well_formed_messages(
    kafka_bootstrap, docs_enriched_topic, tmp_path
):
    from pipeline_v2.messaging import AckAction, ShareGroupQueue

    n = 3
    for i in range(n):
        (tmp_path / f"doc{i}.pdf").write_bytes(f"pdf-bytes-{i}".encode())

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "enqueue",
            "--corpus",
            "it-corpus",
            "--pdf-glob",
            str(tmp_path / "*.pdf"),
            "--bootstrap-servers",
            kafka_bootstrap,
            "--topic",
            docs_enriched_topic,
        ],
    )
    assert result.exit_code == 0, result.output
    assert f"enqueued {n} message" in result.output

    consumed: list[DocsEnrichedMessage] = []
    queue = ShareGroupQueue(
        bootstrap_servers=kafka_bootstrap,
        group_id=f"it-enqueue-check-{uuid.uuid4().hex[:8]}",
        topics=[docs_enriched_topic],
    )
    try:
        for _ in range(10):
            batch = queue.poll_batch(timeout=2.0)
            for qmsg in batch:
                consumed.append(DocsEnrichedMessage.model_validate_json(qmsg.value))
                queue.ack(qmsg, AckAction.ACCEPT)
            if batch:
                queue.commit()
            if len(consumed) >= n:
                break
    finally:
        queue.close()

    assert len(consumed) == n
    assert {m.doc_id for m in consumed} == {f"doc{i}" for i in range(n)}
    for message in consumed:
        assert message.corpus == "it-corpus"
        assert message.stage_fingerprint == cli_module.DEFAULT_ENQUEUE_STAGE_FINGERPRINT
        assert message.payload["pdf_path"].endswith(".pdf")
        assert len(message.content_sha256) == 64
