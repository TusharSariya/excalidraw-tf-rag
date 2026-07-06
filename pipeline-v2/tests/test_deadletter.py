from __future__ import annotations

import json

from fake_queue import FakeWorkQueue
from pipeline_v2.contracts import StageMessage
from pipeline_v2.deadletter import DEFAULT_THRESHOLD, DeadLetterProducer
from pipeline_v2.workstate import InMemoryWorkStateRepository


def test_should_deadletter_threshold_default_is_four():
    assert DEFAULT_THRESHOLD == 4
    dlq = DeadLetterProducer(queue=FakeWorkQueue(), repo=InMemoryWorkStateRepository())
    assert dlq.should_deadletter(1) is False
    assert dlq.should_deadletter(3) is False
    assert dlq.should_deadletter(4) is True
    assert dlq.should_deadletter(5) is True


def test_send_produces_to_deadletter_topic_and_records_failed_outcome():
    queue = FakeWorkQueue()
    repo = InMemoryWorkStateRepository()
    dlq = DeadLetterProducer(queue=queue, repo=repo)

    message = StageMessage(
        doc_id="d1", corpus="glr", content_sha256="sha1", stage_fingerprint="fp1", attempt=4
    )
    dlq.send(
        stage="embed",
        original_topic="chunks.ready",
        reason="handler raised RuntimeError",
        delivery_count=4,
        message=message,
    )

    assert len(queue.produced) == 1
    topic, value = queue.produced[0]
    assert topic == "deadletter.embed"
    body = json.loads(value)
    assert body["doc_id"] == "d1"
    assert body["reason"] == "handler raised RuntimeError"
    assert body["original_topic"] == "chunks.ready"
    assert body["stage"] == "embed"

    row = repo.get("d1", "glr", "embed")
    assert row is not None
    assert row.state == "failed"
    assert row.reason == "handler raised RuntimeError"


def test_send_falls_back_to_unknown_identity_for_poison_with_no_recoverable_fields():
    queue = FakeWorkQueue()
    repo = InMemoryWorkStateRepository()
    dlq = DeadLetterProducer(queue=queue, repo=repo)

    dlq.send(
        stage="extract",
        original_topic="docs.enriched",
        reason="invalid JSON",
        delivery_count=1,
        identity=None,
    )

    topic, value = queue.produced[0]
    body = json.loads(value)
    assert body["doc_id"] == "unknown"
    assert body["corpus"] == "unknown"

    row = repo.get("unknown", "unknown", "extract")
    assert row is not None
    assert row.state == "failed"


def test_send_uses_partial_identity_when_available():
    queue = FakeWorkQueue()
    repo = InMemoryWorkStateRepository()
    dlq = DeadLetterProducer(queue=queue, repo=repo)

    dlq.send(
        stage="extract",
        original_topic="docs.enriched",
        reason="missing required field",
        delivery_count=1,
        identity={"doc_id": "d2", "corpus": "glr"},
    )

    row = repo.get("d2", "glr", "extract")
    assert row is not None
    assert row.state == "failed"
    assert row.content_sha256 == "unknown"
