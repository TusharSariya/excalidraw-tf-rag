from __future__ import annotations

import json

import pytest
from fake_queue import FakeWorkQueue
from pipeline_v2.consumers import (
    HandlerResult,
    IndexerGateError,
    StageConfig,
    StageConsumer,
    make_indexer_handler,
)
from pipeline_v2.contracts import StageMessage
from pipeline_v2.deadletter import DeadLetterProducer
from pipeline_v2.workstate import InMemoryWorkStateRepository


def _msg_bytes(**overrides) -> bytes:
    data = {
        "doc_id": "d1",
        "corpus": "glr",
        "content_sha256": "sha1",
        "stage_fingerprint": "fp1",
        "attempt": 1,
        "payload": {},
    }
    data.update(overrides)
    return json.dumps(data).encode()


def _build(handler, input_topic="chunks.ready", output_topic="chunks.embedded", threshold=4):
    queue = FakeWorkQueue()
    repo = InMemoryWorkStateRepository()
    dlq = DeadLetterProducer(queue=queue, repo=repo, threshold=threshold)
    config = StageConfig(
        name="embed", input_topic=input_topic, handler=handler, output_topic=output_topic
    )
    consumer = StageConsumer(queue=queue, repo=repo, config=config, dlq=dlq, run_id="run-1")
    return queue, repo, consumer


def test_happy_path_ok_produces_next_message_and_ok_row():
    def handler(message: StageMessage) -> HandlerResult:
        return HandlerResult(state="ok", output_payload={"chunks": 3})

    queue, repo, consumer = _build(handler)
    queue.seed("chunks.ready", _msg_bytes())

    consumer.run_once()

    row = repo.get("d1", "glr", "embed")
    assert row is not None
    assert row.state == "ok"
    assert len(queue.produced) == 1
    topic, value = queue.produced[0]
    assert topic == "chunks.embedded"
    assert json.loads(value)["payload"] == {"chunks": 3}
    # record fully acked (no longer in-flight, not re-queued)
    assert list(queue.topics["chunks.ready"]) == []


def test_degraded_outcome_records_row_but_does_not_produce_next_message():
    def handler(message: StageMessage) -> HandlerResult:
        return HandlerResult(state="degraded", reason="ocr fallback")

    queue, repo, consumer = _build(handler)
    queue.seed("chunks.ready", _msg_bytes())

    consumer.run_once()

    row = repo.get("d1", "glr", "embed")
    assert row is not None
    assert row.state == "degraded"
    assert queue.produced == []


def test_poison_message_goes_to_deadletter_immediately_regardless_of_delivery_count():
    def handler(message: StageMessage) -> HandlerResult:
        raise AssertionError("handler must not be called for a poison message")

    queue, repo, consumer = _build(handler)
    queue.seed("chunks.ready", b"not valid json{{{", delivery_count=1)

    consumer.run_once()

    assert len(queue.produced) == 1
    topic, value = queue.produced[0]
    assert topic == "deadletter.embed"
    body = json.loads(value)
    assert "poison" in body["reason"]
    row = repo.get("unknown", "unknown", "embed")
    assert row is not None
    assert row.state == "failed"


def test_failed_below_threshold_is_released_for_redelivery():
    def handler(message: StageMessage) -> HandlerResult:
        return HandlerResult(state="failed", reason="transient error")

    queue, repo, consumer = _build(handler, threshold=4)
    queue.seed("chunks.ready", _msg_bytes(), delivery_count=1)

    consumer.run_once()

    assert queue.produced == []  # not deadlettered yet
    # released -> requeued at front with delivery_count bumped
    assert len(queue.topics["chunks.ready"]) == 1
    assert queue.topics["chunks.ready"][0].delivery_count == 2
    row = repo.get("d1", "glr", "embed")
    assert row is not None
    assert row.state == "failed"


def test_delivery_count_exhaustion_deadletters_and_accepts_original():
    def handler(message: StageMessage) -> HandlerResult:
        return HandlerResult(state="failed", reason="still broken")

    queue, repo, consumer = _build(handler, threshold=4)
    queue.seed("chunks.ready", _msg_bytes(), delivery_count=4)

    consumer.run_once()

    assert len(queue.produced) == 1
    assert queue.produced[0][0] == "deadletter.embed"
    # archived, not requeued
    assert list(queue.topics["chunks.ready"]) == []
    row = repo.get("d1", "glr", "embed")
    assert row is not None
    assert row.state == "failed"


def test_handler_exception_is_treated_as_failed_outcome():
    def handler(message: StageMessage) -> HandlerResult:
        raise RuntimeError("boom")

    queue, repo, consumer = _build(handler, threshold=4)
    queue.seed("chunks.ready", _msg_bytes(), delivery_count=1)

    consumer.run_once()

    row = repo.get("d1", "glr", "embed")
    assert row is not None
    assert row.state == "failed"
    assert "boom" in (row.reason or "")
    assert queue.topics["chunks.ready"][0].delivery_count == 2


def test_idempotent_duplicate_is_skipped_without_calling_handler_or_producing_again():
    calls = []

    def handler(message: StageMessage) -> HandlerResult:
        calls.append(message.doc_id)
        return HandlerResult(state="ok", output_payload={"n": len(calls)})

    queue, repo, consumer = _build(handler)
    queue.seed("chunks.ready", _msg_bytes())
    queue.seed("chunks.ready", _msg_bytes())  # exact duplicate

    consumer.run_once()
    consumer.run_once()

    assert calls == ["d1"]  # handler only invoked once
    assert len(queue.produced) == 1  # only one downstream message


def test_indexer_gate_refuses_when_upstream_row_missing():
    repo = InMemoryWorkStateRepository()
    handler = make_indexer_handler(repo, gate_stage="embed")
    message = StageMessage(
        doc_id="d1", corpus="glr", content_sha256="sha1", stage_fingerprint="fp1"
    )

    with pytest.raises(IndexerGateError):
        handler(message)


def test_indexer_gate_refuses_when_upstream_row_not_ok():
    repo = InMemoryWorkStateRepository()
    repo.upsert_outcome(
        doc_id="d1",
        corpus="glr",
        stage="embed",
        content_sha256="sha1",
        stage_fingerprint="fp1",
        state="degraded",
    )
    handler = make_indexer_handler(repo, gate_stage="embed")
    message = StageMessage(
        doc_id="d1", corpus="glr", content_sha256="sha1", stage_fingerprint="fp1"
    )

    with pytest.raises(IndexerGateError):
        handler(message)


def test_indexer_gate_passes_then_delegates_to_index_writer():
    repo = InMemoryWorkStateRepository()
    repo.upsert_outcome(
        doc_id="d1",
        corpus="glr",
        stage="embed",
        content_sha256="sha1",
        stage_fingerprint="fp1",
        state="ok",
    )
    calls = []

    def fake_writer(message: StageMessage) -> HandlerResult:
        calls.append(message.doc_id)
        return HandlerResult(state="ok", output_payload={"indexed": True})

    handler = make_indexer_handler(repo, gate_stage="embed", index_writer=fake_writer)
    message = StageMessage(
        doc_id="d1", corpus="glr", content_sha256="sha1", stage_fingerprint="fp1"
    )

    result = handler(message)

    assert calls == ["d1"]
    assert result.state == "ok"
    assert result.output_payload == {"indexed": True}


def test_indexer_gate_refuses_stale_message_with_mismatched_fingerprint():
    repo = InMemoryWorkStateRepository()
    repo.upsert_outcome(
        doc_id="d1",
        corpus="glr",
        stage="embed",
        content_sha256="sha1",
        stage_fingerprint="fp1",
        state="ok",
    )
    handler = make_indexer_handler(repo, gate_stage="embed")
    stale_message = StageMessage(
        doc_id="d1", corpus="glr", content_sha256="sha1", stage_fingerprint="fp-STALE"
    )

    with pytest.raises(IndexerGateError):
        handler(stale_message)


def test_extract_stage_config_defaults_to_real_handler_and_degrades_missing_pdf_path():
    from pipeline_v2.consumers import extract_stage_config

    config = extract_stage_config()
    assert config.output_topic == "chunks.ready"
    result = config.handler(
        StageMessage(doc_id="d", corpus="c", content_sha256="s", stage_fingerprint="f")
    )
    assert result.state == "degraded"
    assert "pdf_path" in (result.reason or "")


def test_embed_stage_config_defaults_to_real_handler_and_degrades_missing_artifact():
    from pipeline_v2.consumers import embed_stage_config

    config = embed_stage_config()
    result = config.handler(
        StageMessage(doc_id="d", corpus="c", content_sha256="s", stage_fingerprint="f")
    )
    assert result.state == "degraded"
    assert "chunks_artifact_uri" in (result.reason or "")


def test_stage_configs_accept_injected_fake_handlers():
    from pipeline_v2.consumers import embed_stage_config, extract_stage_config

    def fake(message: StageMessage) -> HandlerResult:
        raise AssertionError("fake handler should not be called in this test")

    assert extract_stage_config(handler=fake).handler is fake
    assert embed_stage_config(handler=fake).handler is fake
