"""Integration tests for the messaging backbone against a real Kafka 4.3.1
broker (KIP-932 share groups) + postgres:16 work_state.

Broker config: see the `kafka_bootstrap` fixture in conftest.py (spike env
verbatim, record lock 600000ms / max 900000ms — see
scratchpad/sharegroup-spike/REPORT.md).

Isolation strategy: the Kafka container is session-scoped (starting a fresh
broker per test would dominate wall-clock time), so each test gets its own
set of real Kafka topics (via `kafka_topics`, suffixed with a UUID) and its
own consumer group id(s), rather than sharing the four fixed topic names
across tests — a fresh share group resets to `earliest` and would otherwise
replay every prior test's messages on the shared topic. `kafka_topics`
temporarily registers the per-test topic names into
`pipeline_v2.contracts.TOPIC_MODELS` (via monkeypatch) and patches
`deadletter_topic` so `consumers.py`/`deadletter.py` — which only know the
fixed topic/stage names — validate and DLQ correctly against the unique
per-test topics.

Every consumer loop below runs with `max_iterations` bounded and a poll
timeout well under a minute total, per the anti-stall discipline in the
task brief: no unbounded poll in the foreground.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest
from pipeline_v2 import contracts as contracts_module
from pipeline_v2 import deadletter as deadletter_module
from pipeline_v2.consumers import HandlerResult, StageConfig, StageConsumer
from pipeline_v2.contracts import ChunksEmbeddedMessage, ChunksReadyMessage, DeadLetterMessage
from pipeline_v2.deadletter import DeadLetterProducer
from pipeline_v2.messaging import AckAction, ClassicGroupQueue, ShareGroupQueue
from pipeline_v2.workstate import PostgresWorkStateRepository

pytestmark = pytest.mark.integration


@pytest.fixture
def kafka_topics(kafka_bootstrap: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    from confluent_kafka.admin import AdminClient, NewTopic

    suffix = uuid.uuid4().hex[:8]
    names = {
        "chunks.ready": f"chunks.ready.{suffix}",
        "chunks.embedded": f"chunks.embedded.{suffix}",
        "deadletter.embed": f"deadletter.embed.{suffix}",
    }

    admin = AdminClient({"bootstrap.servers": kafka_bootstrap})
    new_topics = [NewTopic(name, num_partitions=1, replication_factor=1) for name in names.values()]
    futures = admin.create_topics(new_topics)
    for _name, future in futures.items():
        try:
            future.result(timeout=20)
        except Exception as exc:  # noqa: BLE001
            if "already exists" not in str(exc).lower():
                raise

    monkeypatch.setitem(contracts_module.TOPIC_MODELS, names["chunks.ready"], ChunksReadyMessage)
    monkeypatch.setitem(
        contracts_module.TOPIC_MODELS, names["chunks.embedded"], ChunksEmbeddedMessage
    )
    monkeypatch.setitem(contracts_module.TOPIC_MODELS, names["deadletter.embed"], DeadLetterMessage)
    monkeypatch.setattr(
        deadletter_module,
        "deadletter_topic",
        lambda stage: names[f"deadletter.{stage}"],
    )
    return names


@pytest.fixture
def pg_repo(clean_postgres_dsn: str) -> PostgresWorkStateRepository:
    repo = PostgresWorkStateRepository(clean_postgres_dsn)
    repo.init_db()
    return repo


def _produce_raw(bootstrap: str, topic: str, value: bytes) -> None:
    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": bootstrap})
    producer.produce(topic, value=value)
    producer.flush(15)


def _msg_bytes(**overrides) -> bytes:
    data = {
        "doc_id": "d1",
        "corpus": "glr",
        "content_sha256": "sha1",
        "stage_fingerprint": "fp1",
        "attempt": 1,
        "payload": {"n": 1},
    }
    data.update(overrides)
    return json.dumps(data).encode()


def _drain_topic(bootstrap: str, topic: str, group: str, *, max_iterations: int = 8) -> list[dict]:
    """Best-effort drain of a topic via a fresh share group. Bounded, ACCEPTs everything."""
    queue = ShareGroupQueue(bootstrap_servers=bootstrap, group_id=group, topics=[topic])
    out: list[dict] = []
    try:
        for _ in range(max_iterations):
            batch = queue.poll_batch(timeout=2.0)
            for qmsg in batch:
                out.append(json.loads(qmsg.value))
                queue.ack(qmsg, AckAction.ACCEPT)
            if batch:
                queue.commit()
            if out:
                break
    finally:
        queue.close()
    return out


def test_happy_path_produce_consume_ack_workstate_ok_and_next_topic_message(
    kafka_bootstrap, kafka_topics, pg_repo
):
    _produce_raw(kafka_bootstrap, kafka_topics["chunks.ready"], _msg_bytes())

    def handler(message):
        return HandlerResult(state="ok", output_payload={"embedded": True})

    config = StageConfig(
        name="embed",
        input_topic=kafka_topics["chunks.ready"],
        handler=handler,
        output_topic=kafka_topics["chunks.embedded"],
    )
    group = f"it-happy-{uuid.uuid4().hex[:8]}"
    queue = ShareGroupQueue(
        bootstrap_servers=kafka_bootstrap, group_id=group, topics=[config.input_topic]
    )
    dlq = DeadLetterProducer(queue=queue, repo=pg_repo)
    consumer = StageConsumer(queue=queue, repo=pg_repo, config=config, dlq=dlq, run_id="it-happy")
    try:
        consumer.run_forever(max_iterations=10, poll_timeout=2.0)
    finally:
        queue.close()

    row = pg_repo.get("d1", "glr", "embed")
    assert row is not None
    assert row.state == "ok"

    downstream = _drain_topic(
        kafka_bootstrap, kafka_topics["chunks.embedded"], f"it-happy-check-{uuid.uuid4().hex[:8]}"
    )
    assert len(downstream) == 1
    assert downstream[0]["doc_id"] == "d1"
    assert downstream[0]["payload"] == {"embedded": True}


def test_poison_message_to_deadletter_topic_and_failed_row(kafka_bootstrap, kafka_topics, pg_repo):
    _produce_raw(kafka_bootstrap, kafka_topics["chunks.ready"], b"not valid json{{{")

    def handler(message):
        raise AssertionError("handler must not run for a poison message")

    config = StageConfig(
        name="embed",
        input_topic=kafka_topics["chunks.ready"],
        handler=handler,
        output_topic=kafka_topics["chunks.embedded"],
    )
    group = f"it-poison-{uuid.uuid4().hex[:8]}"
    queue = ShareGroupQueue(
        bootstrap_servers=kafka_bootstrap, group_id=group, topics=[config.input_topic]
    )
    dlq = DeadLetterProducer(queue=queue, repo=pg_repo)
    consumer = StageConsumer(queue=queue, repo=pg_repo, config=config, dlq=dlq, run_id="it-poison")
    try:
        consumer.run_forever(max_iterations=10, poll_timeout=2.0)
    finally:
        queue.close()

    dlq_messages = _drain_topic(
        kafka_bootstrap, kafka_topics["deadletter.embed"], f"it-poison-check-{uuid.uuid4().hex[:8]}"
    )
    assert len(dlq_messages) == 1
    assert "poison" in dlq_messages[0]["reason"]

    row = pg_repo.get("unknown", "unknown", "embed")
    assert row is not None
    assert row.state == "failed"


def test_crash_redelivery_second_consumer_sees_delivery_count_two(kafka_bootstrap, kafka_topics):
    import subprocess
    import sys
    from pathlib import Path

    topic = kafka_topics["chunks.ready"]
    group = f"it-crash-{uuid.uuid4().hex[:8]}"
    _produce_raw(kafka_bootstrap, topic, _msg_bytes(doc_id="crash-doc"))

    worker = Path(__file__).parent / "_share_crash_worker.py"
    result = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--bootstrap",
            kafka_bootstrap,
            "--topic",
            topic,
            "--group",
            group,
            "--timeout",
            "20",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, f"crash worker stdout={result.stdout} stderr={result.stderr}"
    first = json.loads(result.stdout.strip().splitlines()[-1])
    assert first["delivery_count"] == 1

    # The broker drops the crashed member's session and releases the record
    # for redelivery well within our poll budget (spike measured ~6s).
    queue = ShareGroupQueue(bootstrap_servers=kafka_bootstrap, group_id=group, topics=[topic])
    try:
        second_delivery_count = None
        for _ in range(10):
            batch = queue.poll_batch(timeout=2.0)
            if batch:
                second_delivery_count = batch[0].delivery_count
                queue.ack(batch[0], AckAction.ACCEPT)
                queue.commit()
                break
    finally:
        queue.close()

    assert second_delivery_count == 2


def test_delivery_count_exhaustion_dlq_and_accept_nothing_lost(
    kafka_bootstrap, kafka_topics, pg_repo
):
    input_topic = kafka_topics["chunks.ready"]
    _produce_raw(kafka_bootstrap, input_topic, _msg_bytes(doc_id="exhaust-doc"))

    def always_fails(message):
        return HandlerResult(state="failed", reason="permanent failure")

    config = StageConfig(
        name="embed", input_topic=input_topic, handler=always_fails, output_topic=None
    )
    group = f"it-exhaust-{uuid.uuid4().hex[:8]}"
    queue = ShareGroupQueue(bootstrap_servers=kafka_bootstrap, group_id=group, topics=[input_topic])
    dlq = DeadLetterProducer(queue=queue, repo=pg_repo, threshold=4)
    consumer = StageConsumer(queue=queue, repo=pg_repo, config=config, dlq=dlq, run_id="it-exhaust")
    try:
        # Each run_once RELEASEs the same record (redelivered with delivery_count+1)
        # until the 4th attempt hits the DLQ threshold and ACCEPTs it. A handful
        # of extra iterations cover poll-timing slack without risking an
        # unbounded loop.
        consumer.run_forever(max_iterations=12, poll_timeout=2.0)
    finally:
        queue.close()

    row = pg_repo.get("exhaust-doc", "glr", "embed")
    assert row is not None
    assert row.state == "failed"

    dlq_messages = _drain_topic(
        kafka_bootstrap,
        kafka_topics["deadletter.embed"],
        f"it-exhaust-check-{uuid.uuid4().hex[:8]}",
    )
    assert len(dlq_messages) == 1
    assert dlq_messages[0]["doc_id"] == "exhaust-doc"
    assert dlq_messages[0]["payload"] == {"n": 1}  # original payload preserved, nothing lost

    # Nothing left for THIS group — the exhausted record was archived out of
    # its share-partition state (a brand-new group would still see it from
    # `earliest`, since archiving only advances this group's start-offset,
    # not the underlying log — so re-check with the *same* group id).
    leftover = _drain_topic(kafka_bootstrap, input_topic, group, max_iterations=3)
    assert leftover == []


def test_idempotent_duplicate_one_row_one_downstream_message(
    kafka_bootstrap, kafka_topics, pg_repo
):
    input_topic = kafka_topics["chunks.ready"]
    body = _msg_bytes(doc_id="dup-doc")
    _produce_raw(kafka_bootstrap, input_topic, body)
    _produce_raw(kafka_bootstrap, input_topic, body)  # exact duplicate

    calls = []

    def handler(message):
        calls.append(message.doc_id)
        return HandlerResult(state="ok", output_payload={"seen": len(calls)})

    config = StageConfig(
        name="embed",
        input_topic=input_topic,
        handler=handler,
        output_topic=kafka_topics["chunks.embedded"],
    )
    group = f"it-dup-{uuid.uuid4().hex[:8]}"
    queue = ShareGroupQueue(bootstrap_servers=kafka_bootstrap, group_id=group, topics=[input_topic])
    dlq = DeadLetterProducer(queue=queue, repo=pg_repo)
    consumer = StageConsumer(queue=queue, repo=pg_repo, config=config, dlq=dlq, run_id="it-dup")
    try:
        consumer.run_forever(max_iterations=10, poll_timeout=2.0)
    finally:
        queue.close()

    row = pg_repo.get("dup-doc", "glr", "embed")
    assert row is not None
    assert row.state == "ok"
    assert calls == ["dup-doc"]  # handler only ran once — second message was an idempotent skip

    downstream = _drain_topic(
        kafka_bootstrap, kafka_topics["chunks.embedded"], f"it-dup-check-{uuid.uuid4().hex[:8]}"
    )
    assert len(downstream) == 1


def test_classic_group_queue_happy_path(kafka_bootstrap, kafka_topics, pg_repo):
    _produce_raw(kafka_bootstrap, kafka_topics["chunks.ready"], _msg_bytes(doc_id="classic-doc"))

    def handler(message):
        return HandlerResult(state="ok", output_payload={"backend": "classic"})

    config = StageConfig(
        name="embed",
        input_topic=kafka_topics["chunks.ready"],
        handler=handler,
        output_topic=kafka_topics["chunks.embedded"],
    )
    group = f"it-classic-{uuid.uuid4().hex[:8]}"
    queue = ClassicGroupQueue(
        bootstrap_servers=kafka_bootstrap, group_id=group, topics=[config.input_topic]
    )
    dlq = DeadLetterProducer(queue=queue, repo=pg_repo)
    consumer = StageConsumer(queue=queue, repo=pg_repo, config=config, dlq=dlq, run_id="it-classic")
    try:
        consumer.run_forever(max_iterations=15, poll_timeout=2.0)
    finally:
        queue.close()

    row = pg_repo.get("classic-doc", "glr", "embed")
    assert row is not None
    assert row.state == "ok"

    # Drain the downstream topic with a classic group too (fallback parity).
    check_queue = ClassicGroupQueue(
        bootstrap_servers=kafka_bootstrap,
        group_id=f"it-classic-check-{uuid.uuid4().hex[:8]}",
        topics=[kafka_topics["chunks.embedded"]],
    )
    downstream: list[dict] = []
    try:
        for _ in range(8):
            batch = check_queue.poll_batch(timeout=2.0)
            for qmsg in batch:
                downstream.append(json.loads(qmsg.value))
                check_queue.ack(qmsg, AckAction.ACCEPT)
            if batch:
                check_queue.commit()
            if downstream:
                break
    finally:
        check_queue.close()

    assert len(downstream) == 1
    assert downstream[0]["payload"] == {"backend": "classic"}


def test_share_group_offset_reset_provisioning_uses_admin_client(kafka_bootstrap, kafka_topics):
    """Smoke test for ensure_share_group_earliest_reset: a group created by
    ShareGroupQueue AFTER a message was already produced still sees it
    (earliest, not latest) — this is the exact spike gotcha (REPORT.md
    Milestone 2): share.auto.offset.reset defaults to latest and is
    captured on first attach, so we must set it before that first attach.
    """
    topic = kafka_topics["chunks.ready"]
    _produce_raw(kafka_bootstrap, topic, _msg_bytes(doc_id="reset-doc"))
    time.sleep(1.0)  # let the produce settle before the group's first attach

    group = f"it-reset-{uuid.uuid4().hex[:8]}"
    queue = ShareGroupQueue(bootstrap_servers=kafka_bootstrap, group_id=group, topics=[topic])
    try:
        seen = None
        for _ in range(8):
            batch = queue.poll_batch(timeout=2.0)
            if batch:
                seen = json.loads(batch[0].value)
                queue.ack(batch[0], AckAction.ACCEPT)
                queue.commit()
                break
        assert seen is not None, "message produced before first attach must still be delivered"
        assert seen["doc_id"] == "reset-doc"
    finally:
        queue.close()
