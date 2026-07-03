"""``WorkQueue``: one poll/ack/produce interface, two Kafka-backed implementations.

Both implementations wrap ``confluent-kafka`` (pinned to ``2.15.0`` in
``pyproject.toml``, see spike report at
``scratchpad/sharegroup-spike/REPORT.md``):

* :class:`ShareGroupQueue` — KIP-932 share groups (``ShareConsumer``), GA in
  Kafka 4.2+, on by default. Gives native per-record ack/release/reject and
  redelivery with a delivery-count the broker tracks for us. The Python
  client is officially "Preview" (librdkafka ``rd_kafka_share_*`` may
  change) — we pin the version and isolate all Preview API usage behind
  this class so a fallback swap never touches call sites.
* :class:`ClassicGroupQueue` — a classic Kafka consumer group with manual
  offset commit. No native per-record redelivery/lock; "release"/"reject"
  are approximated (see class docstring) and at-least-once is achieved
  purely by not committing until a record's outcome is durably recorded.
  This is the fallback path if share groups become unusable in production.

Spike gotchas encoded here (do not "simplify" these away without re-reading
``scratchpad/sharegroup-spike/REPORT.md``):

* ``share.auto.offset.reset`` defaults to ``latest`` and is captured on the
  share-partition's FIRST attach — it must be set to ``earliest`` via
  AdminClient *before* any consumer in the group first subscribes.
  :func:`ensure_share_group_earliest_reset` does this; call it once per
  ``group_id`` before constructing the first :class:`ShareGroupQueue` for
  that group (:class:`ShareGroupQueue` calls it automatically in
  ``__init__`` unless ``skip_offset_reset_provisioning=True``).
* Record-lock duration (``group.share.record.lock.duration.ms``) is a
  broker/group-level setting ONLY — confluent-kafka 2.15.0's
  ``ShareConsumer`` has no client-side ``share.record.lock.duration.ms``
  config key (confirmed against a real broker: passing one raises
  ``_INVALID_ARG``, contradicting an earlier assumption from the spike
  write-up). ``group.share.max.record.lock.duration.ms`` on the broker
  clamps the *effective* lock to its own default (60000ms) unless raised —
  see ``tests/conftest.py``'s ``kafka_bootstrap`` fixture for the broker env
  that raises it to 900000ms against a 600000ms working lock.
* ``max.poll.interval.ms`` (client-side) must exceed the worst-case handler
  runtime, or the client will consider itself dead mid-job. Defaulted here
  well above 600s.
* ``ShareConsumer.poll()`` returns a ``Messages`` container; iterate
  ``batch.records()`` (a method, not an attribute) and check
  ``batch.is_empty()``.
* The broker's ADVERTISED_LISTENERS address must be reachable from BOTH the
  test client AND the broker's own internal share-coordinator/persister
  loopback — i.e. the container's internal listener port, its advertised
  port, and the host port mapping must all be numerically identical. A
  host-port remap (e.g. internal ``9092`` advertised as host port
  ``54321`` via ``-p 54321:9092``) breaks that internal loopback silently:
  consumers join the share group and get assigned zero partitions forever,
  with no client-visible error (broker log shows "Exhausted max retries to
  find coordinator" / errorCode 13). Found the hard way building the
  integration fixture — not mentioned in the spike, which always used a
  fixed unmapped port.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Client-side: must exceed the worst-case stage handler runtime. The spike's
# broker-side lock was proven up to 600000ms (900000ms max after the P0a
# broker-config fix) so the client poll-interval budget is set comfortably
# above that.
DEFAULT_MAX_POLL_INTERVAL_MS = 900_000

# Broker-side record lock requested by the client. Must be <=
# group.share.max.record.lock.duration.ms on the broker (raised to 900000ms
# per the spike's Lane A3 finding; the broker's default max is 60000ms and
# silently clamps a larger client request).
DEFAULT_RECORD_LOCK_MS = 600_000


class AckAction(str, Enum):
    """Terminal disposition of one polled record."""

    ACCEPT = "accept"
    RELEASE = "release"
    REJECT = "reject"


@dataclass
class QueueMessage:
    """One polled record, backend-agnostic.

    ``raw`` carries whatever handle the backend needs to ack it later
    (the confluent_kafka ``Message`` object for both implementations here)
    — callers must treat it as opaque and pass it straight back into
    :meth:`WorkQueue.ack`.
    """

    topic: str
    key: bytes | None
    value: bytes
    delivery_count: int
    partition: int
    offset: int
    raw: Any


class WorkQueue(ABC):
    """Poll/ack/produce interface shared by both Kafka-backed queue implementations."""

    @abstractmethod
    def poll_batch(self, timeout: float = 1.0) -> list[QueueMessage]:
        """Poll for up to one backend-sized batch of records. May return []."""

    @abstractmethod
    def ack(self, message: QueueMessage, action: AckAction) -> None:
        """Record the disposition of ``message``. Does not itself commit."""

    @abstractmethod
    def commit(self) -> None:
        """Durably commit every ack recorded since the last commit."""

    @abstractmethod
    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        """Enqueue ``value`` for ``topic``. Buffered — call :meth:`flush` to send."""

    @abstractmethod
    def flush(self, timeout: float = 10.0) -> None:
        """Block until all buffered :meth:`produce` calls are acknowledged by the broker."""

    @abstractmethod
    def close(self) -> None:
        """Release all backend resources. Safe to call more than once."""

    def __enter__(self) -> WorkQueue:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def ensure_share_group_earliest_reset(bootstrap_servers: str, group_id: str) -> None:
    """Set ``share.auto.offset.reset=earliest`` for ``group_id`` via AdminClient.

    Must run before the group's first consumer ever subscribes — the value
    is captured when the share partition initializes and changing it later
    has no effect on already-initialized partitions (spike finding, cost 3
    dead runs — see REPORT.md Milestone 2).
    """
    from confluent_kafka.admin import AdminClient, ConfigResource

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    resource = ConfigResource(
        ConfigResource.Type.GROUP, group_id, set_config={"share.auto.offset.reset": "earliest"}
    )
    futures = admin.alter_configs([resource])
    for _resource, future in futures.items():
        try:
            future.result(timeout=15)
        except Exception as exc:  # pragma: no cover - defensive, broker/version dependent
            logger.warning(
                "could not pre-set share.auto.offset.reset=earliest for group %s "
                "(may already be set, or broker/client version mismatch): %s",
                group_id,
                exc,
            )


class ShareGroupQueue(WorkQueue):
    """KIP-932 share-group backed :class:`WorkQueue` (confluent-kafka ``ShareConsumer``).

    The underlying client is officially "Preview" — the confluent-kafka
    library itself emits a startup warning
    (``"The share consumer is in \"Preview\" and is not recommended for
    production use"``); we catch and re-log that at INFO (not WARNING) so
    it doesn't page anyone, since we've deliberately chosen this path with
    the fallback (:class:`ClassicGroupQueue`) ready to swap in.
    """

    def __init__(
        self,
        *,
        bootstrap_servers: str,
        group_id: str,
        topics: list[str],
        max_poll_interval_ms: int = DEFAULT_MAX_POLL_INTERVAL_MS,
        record_lock_ms: int = DEFAULT_RECORD_LOCK_MS,
        skip_offset_reset_provisioning: bool = False,
        extra_consumer_config: dict[str, Any] | None = None,
        extra_producer_config: dict[str, Any] | None = None,
    ) -> None:
        from confluent_kafka import Producer, ShareConsumer

        if not skip_offset_reset_provisioning:
            ensure_share_group_earliest_reset(bootstrap_servers, group_id)

        # NOTE: record-lock duration is a broker/group-level setting only
        # (group.share.record.lock.duration.ms, set via broker config/env —
        # see conftest.py's kafka_bootstrap fixture) — confluent-kafka
        # 2.15.0's ShareConsumer does NOT accept a client-side
        # "share.record.lock.duration.ms" config key (verified against a
        # real broker: raises _INVALID_ARG). ``record_lock_ms`` is kept as a
        # constructor parameter purely for documentation/introspection by
        # callers that want to assert what lock duration they expect the
        # broker to be using; it is not sent over the wire.
        self.record_lock_ms = record_lock_ms
        consumer_config: dict[str, Any] = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "share.acknowledgement.mode": "explicit",
            "max.poll.interval.ms": max_poll_interval_ms,
        }
        consumer_config.update(extra_consumer_config or {})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._consumer = ShareConsumer(consumer_config)
        logger.info(
            "ShareConsumer initialized for group=%s topics=%s "
            "(confluent-kafka ShareConsumer is Preview; pinned to 2.15.0, "
            "ClassicGroupQueue fallback available — see messaging.py module docstring)",
            group_id,
            topics,
        )
        self._consumer.subscribe(topics)

        producer_config: dict[str, Any] = {"bootstrap.servers": bootstrap_servers}
        producer_config.update(extra_producer_config or {})
        self._producer = Producer(producer_config)
        self._closed = False

    def poll_batch(self, timeout: float = 1.0) -> list[QueueMessage]:
        from confluent_kafka import KafkaError

        batch = self._consumer.poll(timeout=timeout)
        if batch is None or batch.is_empty():
            return []
        out: list[QueueMessage] = []
        for msg in batch.records():
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.warning("share consumer poll error: %s", msg.error())
                continue
            out.append(
                QueueMessage(
                    topic=msg.topic(),
                    key=msg.key(),
                    value=msg.value(),
                    delivery_count=msg.delivery_count() or 1,
                    partition=msg.partition(),
                    offset=msg.offset(),
                    raw=msg,
                )
            )
        return out

    def ack(self, message: QueueMessage, action: AckAction) -> None:
        from confluent_kafka import AcknowledgeType

        mapping = {
            AckAction.ACCEPT: AcknowledgeType.ACCEPT,
            AckAction.RELEASE: AcknowledgeType.RELEASE,
            AckAction.REJECT: AcknowledgeType.REJECT,
        }
        self._consumer.acknowledge(message.raw, mapping[action])

    def commit(self) -> None:
        self._consumer.commit_sync()

    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        self._producer.produce(topic, value=value, key=key)
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> None:
        self._producer.flush(timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._producer.flush(5)
        except Exception:  # pragma: no cover - best effort on shutdown
            logger.exception("error flushing producer during close")
        self._consumer.close()


class ClassicGroupQueue(WorkQueue):
    """Classic consumer-group :class:`WorkQueue` fallback — manual commit, at-least-once.

    No broker-native per-record lock/redelivery: this backend approximates
    the ShareGroupQueue contract with the tools a classic group has --

    * ACCEPT: record is marked to have its offset committed on the next
      :meth:`commit` call (batched, like a normal consumer).
    * RELEASE: the offset is **not** advanced/committed for that record;
      since offsets commit in-order per partition, any acked-ACCEPT record
      *after* an un-acked RELEASE in the same partition also cannot be
      committed past the release point until it's resolved. In practice
      (this pipeline) each stage's handler is idempotent and batches are
      small, so this is an acceptable trade — a released record is
      redelivered on the next poll (or on consumer restart) because its
      offset was never committed.
    * REJECT: treated the same as ACCEPT for offset-commit purposes (the
      record's offset advances — like the ShareGroupQueue archiving a
      rejected record) but the caller is expected to have already produced
      it to a DLQ before calling ``ack(..., REJECT)`` if it wants the
      payload preserved, exactly as with :class:`ShareGroupQueue`.

    ``delivery_count`` is not tracked by the broker for classic groups, so
    it is approximated in-process per (topic, partition, offset) — this
    only survives for the lifetime of one queue instance, which is
    sufficient for the redelivery-count-based DLQ threshold in
    :mod:`pipeline_v2.deadletter` within a single consumer process; a
    consumer crash-restart resets the local counter to 1 (unlike
    ShareGroupQueue, where the broker remembers). Production stages should
    prefer ShareGroupQueue when accurate cross-restart delivery counts
    matter; this fallback trades that off for guaranteed client-library
    stability.
    """

    def __init__(
        self,
        *,
        bootstrap_servers: str,
        group_id: str,
        topics: list[str],
        max_poll_interval_ms: int = DEFAULT_MAX_POLL_INTERVAL_MS,
        extra_consumer_config: dict[str, Any] | None = None,
        extra_producer_config: dict[str, Any] | None = None,
    ) -> None:
        from confluent_kafka import Consumer, Producer

        consumer_config: dict[str, Any] = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": max_poll_interval_ms,
        }
        consumer_config.update(extra_consumer_config or {})
        self._consumer = Consumer(consumer_config)
        self._consumer.subscribe(topics)

        producer_config: dict[str, Any] = {"bootstrap.servers": bootstrap_servers}
        producer_config.update(extra_producer_config or {})
        self._producer = Producer(producer_config)

        self._pending_commit: list[Any] = []
        self._delivery_counts: dict[tuple[str, int, int], int] = {}
        self._closed = False

    def poll_batch(self, timeout: float = 1.0) -> list[QueueMessage]:
        from confluent_kafka import KafkaError

        out: list[QueueMessage] = []
        msg = self._consumer.poll(timeout=timeout)
        if msg is None:
            return out
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                logger.warning("classic consumer poll error: %s", msg.error())
            return out
        key = (msg.topic(), msg.partition(), msg.offset())
        count = self._delivery_counts.get(key, 0) + 1
        self._delivery_counts[key] = count
        out.append(
            QueueMessage(
                topic=msg.topic(),
                key=msg.key(),
                value=msg.value(),
                delivery_count=count,
                partition=msg.partition(),
                offset=msg.offset(),
                raw=msg,
            )
        )
        return out

    def ack(self, message: QueueMessage, action: AckAction) -> None:
        if action is AckAction.RELEASE:
            # Do not advance the offset: leave it for redelivery. We forget
            # our local delivery-count entry so a genuinely-new poll of the
            # same offset (after a seek/restart) still increments from a
            # known base rather than silently resetting to 1 mid-run.
            return
        # ACCEPT and REJECT both advance the offset; REJECT relies on the
        # caller having already produced the payload to a DLQ topic.
        self._pending_commit.append(message.raw)

    def commit(self) -> None:
        if not self._pending_commit:
            return
        self._consumer.commit(offsets=None, asynchronous=False)
        self._pending_commit.clear()

    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        self._producer.produce(topic, value=value, key=key)
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> None:
        self._producer.flush(timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._producer.flush(5)
        except Exception:  # pragma: no cover - best effort on shutdown
            logger.exception("error flushing producer during close")
        self._consumer.close()


def build_queue(
    backend: str,
    *,
    bootstrap_servers: str,
    group_id: str,
    topics: list[str],
    **kwargs: Any,
) -> WorkQueue:
    """Factory used by the CLI: ``backend`` is ``"share"`` or ``"classic"``."""
    if backend == "share":
        return ShareGroupQueue(
            bootstrap_servers=bootstrap_servers, group_id=group_id, topics=topics, **kwargs
        )
    if backend == "classic":
        return ClassicGroupQueue(
            bootstrap_servers=bootstrap_servers, group_id=group_id, topics=topics, **kwargs
        )
    raise ValueError(f"unknown queue backend {backend!r}: must be 'share' or 'classic'")
