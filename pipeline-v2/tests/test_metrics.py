"""Tests for pipeline_v2.metrics counters/histogram wired into StageConsumer (T0f).

The Prometheus Counter/Histogram objects in ``pipeline_v2.metrics`` are
process-global (module-level singletons, as ``prometheus_client`` requires),
so each test uses its own unique ``corpus`` label value to avoid cross-test
interference, and reads sample values directly off the label child
(``.labels(...)._value.get()`` / ``._sum.get()`` / ``._count.get()``) rather
than binding a real HTTP port -- these tests never call
``start_metrics_server``.
"""

from __future__ import annotations

import json
import uuid

from fake_queue import FakeWorkQueue
from pipeline_v2.consumers import HandlerResult, StageConfig, StageConsumer
from pipeline_v2.contracts import StageMessage
from pipeline_v2.deadletter import DeadLetterProducer
from pipeline_v2.metrics import (
    HANDLER_SECONDS,
    MESSAGES_ACKED_TOTAL,
    MESSAGES_DLQ_TOTAL,
    MESSAGES_PROCESSED_TOTAL,
    MESSAGES_RELEASED_TOTAL,
)
from pipeline_v2.workstate import InMemoryWorkStateRepository


def _msg_bytes(*, corpus: str, **overrides) -> bytes:
    data = {
        "doc_id": "d1",
        "corpus": corpus,
        "content_sha256": "sha1",
        "stage_fingerprint": "fp1",
        "attempt": 1,
        "payload": {},
    }
    data.update(overrides)
    return json.dumps(data).encode()


def _build(handler, *, stage="embed", threshold=4):
    queue = FakeWorkQueue()
    repo = InMemoryWorkStateRepository()
    dlq = DeadLetterProducer(queue=queue, repo=repo, threshold=threshold)
    config = StageConfig(
        name=stage, input_topic="chunks.ready", handler=handler, output_topic="chunks.embedded"
    )
    consumer = StageConsumer(queue=queue, repo=repo, config=config, dlq=dlq, run_id="run-1")
    return queue, repo, consumer


def _counter_value(counter, *, stage: str, corpus: str) -> float:
    return counter.labels(stage=stage, corpus=corpus)._value.get()


def _histogram_count(*, stage: str, corpus: str) -> float:
    for metric in HANDLER_SECONDS.collect():
        for sample in metric.samples:
            if (
                sample.name.endswith("_count")
                and sample.labels.get("stage") == stage
                and sample.labels.get("corpus") == corpus
            ):
                return sample.value
    return 0.0


def test_ok_outcome_increments_processed_and_acked_and_observes_handler_seconds():
    corpus = f"c-{uuid.uuid4().hex[:8]}"

    def handler(message: StageMessage) -> HandlerResult:
        return HandlerResult(state="ok", output_payload={})

    queue, repo, consumer = _build(handler)
    queue.seed("chunks.ready", _msg_bytes(corpus=corpus))

    consumer.run_once()

    assert _counter_value(MESSAGES_PROCESSED_TOTAL, stage="embed", corpus=corpus) == 1
    assert _counter_value(MESSAGES_ACKED_TOTAL, stage="embed", corpus=corpus) == 1
    assert _counter_value(MESSAGES_RELEASED_TOTAL, stage="embed", corpus=corpus) == 0
    assert _counter_value(MESSAGES_DLQ_TOTAL, stage="embed", corpus=corpus) == 0
    assert _histogram_count(stage="embed", corpus=corpus) == 1


def test_degraded_outcome_increments_processed_and_acked_not_released_or_dlq():
    corpus = f"c-{uuid.uuid4().hex[:8]}"

    def handler(message: StageMessage) -> HandlerResult:
        return HandlerResult(state="degraded", reason="ocr fallback")

    queue, repo, consumer = _build(handler)
    queue.seed("chunks.ready", _msg_bytes(corpus=corpus))

    consumer.run_once()

    assert _counter_value(MESSAGES_PROCESSED_TOTAL, stage="embed", corpus=corpus) == 1
    assert _counter_value(MESSAGES_ACKED_TOTAL, stage="embed", corpus=corpus) == 1
    assert _counter_value(MESSAGES_RELEASED_TOTAL, stage="embed", corpus=corpus) == 0
    assert _counter_value(MESSAGES_DLQ_TOTAL, stage="embed", corpus=corpus) == 0
    assert _histogram_count(stage="embed", corpus=corpus) == 1


def test_failed_below_threshold_increments_processed_and_released_not_acked_or_dlq():
    corpus = f"c-{uuid.uuid4().hex[:8]}"

    def handler(message: StageMessage) -> HandlerResult:
        return HandlerResult(state="failed", reason="transient")

    queue, repo, consumer = _build(handler, threshold=4)
    queue.seed("chunks.ready", _msg_bytes(corpus=corpus), delivery_count=1)

    consumer.run_once()

    assert _counter_value(MESSAGES_PROCESSED_TOTAL, stage="embed", corpus=corpus) == 1
    assert _counter_value(MESSAGES_RELEASED_TOTAL, stage="embed", corpus=corpus) == 1
    assert _counter_value(MESSAGES_ACKED_TOTAL, stage="embed", corpus=corpus) == 0
    assert _counter_value(MESSAGES_DLQ_TOTAL, stage="embed", corpus=corpus) == 0
    assert _histogram_count(stage="embed", corpus=corpus) == 1


def test_failed_at_threshold_deadletters_and_increments_dlq_and_acked():
    corpus = f"c-{uuid.uuid4().hex[:8]}"

    def handler(message: StageMessage) -> HandlerResult:
        return HandlerResult(state="failed", reason="still broken")

    queue, repo, consumer = _build(handler, threshold=4)
    queue.seed("chunks.ready", _msg_bytes(corpus=corpus), delivery_count=4)

    consumer.run_once()

    assert _counter_value(MESSAGES_PROCESSED_TOTAL, stage="embed", corpus=corpus) == 1
    assert _counter_value(MESSAGES_DLQ_TOTAL, stage="embed", corpus=corpus) == 1
    assert _counter_value(MESSAGES_ACKED_TOTAL, stage="embed", corpus=corpus) == 1
    assert _counter_value(MESSAGES_RELEASED_TOTAL, stage="embed", corpus=corpus) == 0
    assert _histogram_count(stage="embed", corpus=corpus) == 1


def test_handler_exception_still_observes_handler_seconds_and_counts_as_processed():
    corpus = f"c-{uuid.uuid4().hex[:8]}"

    def handler(message: StageMessage) -> HandlerResult:
        raise RuntimeError("boom")

    queue, repo, consumer = _build(handler, threshold=4)
    queue.seed("chunks.ready", _msg_bytes(corpus=corpus), delivery_count=1)

    consumer.run_once()

    assert _counter_value(MESSAGES_PROCESSED_TOTAL, stage="embed", corpus=corpus) == 1
    assert _histogram_count(stage="embed", corpus=corpus) == 1
    assert _counter_value(MESSAGES_RELEASED_TOTAL, stage="embed", corpus=corpus) == 1


def test_poison_message_increments_dlq_and_acked_but_not_processed_or_histogram():
    corpus_label = "unknown"

    def handler(message: StageMessage):
        raise AssertionError("handler must not be called for a poison message")

    queue, repo, consumer = _build(handler)
    before_processed = _counter_value(MESSAGES_PROCESSED_TOTAL, stage="embed", corpus=corpus_label)
    before_dlq = _counter_value(MESSAGES_DLQ_TOTAL, stage="embed", corpus=corpus_label)
    before_acked = _counter_value(MESSAGES_ACKED_TOTAL, stage="embed", corpus=corpus_label)

    queue.seed("chunks.ready", b"not valid json{{{", delivery_count=1)
    consumer.run_once()

    assert (
        _counter_value(MESSAGES_PROCESSED_TOTAL, stage="embed", corpus=corpus_label)
        == before_processed
    )
    assert _counter_value(MESSAGES_DLQ_TOTAL, stage="embed", corpus=corpus_label) == before_dlq + 1
    assert (
        _counter_value(MESSAGES_ACKED_TOTAL, stage="embed", corpus=corpus_label)
        == before_acked + 1
    )


def test_idempotent_duplicate_does_not_increment_processed_again():
    corpus = f"c-{uuid.uuid4().hex[:8]}"
    calls = []

    def handler(message: StageMessage) -> HandlerResult:
        calls.append(message.doc_id)
        return HandlerResult(state="ok", output_payload={})

    queue, repo, consumer = _build(handler)
    queue.seed("chunks.ready", _msg_bytes(corpus=corpus))
    queue.seed("chunks.ready", _msg_bytes(corpus=corpus))  # exact duplicate

    consumer.run_once()
    consumer.run_once()

    assert calls == ["d1"]
    assert _counter_value(MESSAGES_PROCESSED_TOTAL, stage="embed", corpus=corpus) == 1
    # both records got ACCEPTed off the queue (the duplicate is an idempotent skip)
    assert _counter_value(MESSAGES_ACKED_TOTAL, stage="embed", corpus=corpus) == 2
