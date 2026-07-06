"""An in-memory WorkQueue used by the offline consumer/DLQ unit tests.

Mimics enough real-queue behavior to exercise StageConsumer/DeadLetterProducer
logic without Kafka: topics are FIFO lists of (value, delivery_count) tuples,
poll_batch pops up to one message, RELEASE re-enqueues at the *front* with an
incremented delivery_count (so the next poll sees it again, count bumped —
same externally-visible behavior as a share group's redelivery), ACCEPT/REJECT
drop it for good.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from pipeline_v2.messaging import AckAction, QueueMessage, WorkQueue


@dataclass
class _Pending:
    topic: str
    value: bytes
    delivery_count: int


class FakeWorkQueue(WorkQueue):
    def __init__(self) -> None:
        self.topics: dict[str, deque[_Pending]] = defaultdict(deque)
        self.produced: list[tuple[str, bytes]] = []
        self._in_flight: dict[int, _Pending] = {}
        self._next_id = 0
        self.closed = False

    def seed(self, topic: str, value: bytes, delivery_count: int = 1) -> None:
        self.topics[topic].append(_Pending(topic=topic, value=value, delivery_count=delivery_count))

    def poll_batch(self, timeout: float = 1.0) -> list[QueueMessage]:
        for topic, queue in self.topics.items():
            if queue:
                pending = queue.popleft()
                mid = self._next_id
                self._next_id += 1
                self._in_flight[mid] = pending
                return [
                    QueueMessage(
                        topic=topic,
                        key=None,
                        value=pending.value,
                        delivery_count=pending.delivery_count,
                        partition=0,
                        offset=mid,
                        raw=mid,
                    )
                ]
        return []

    def ack(self, message: QueueMessage, action: AckAction) -> None:
        pending = self._in_flight.pop(message.raw)
        if action is AckAction.RELEASE:
            pending.delivery_count += 1
            self.topics[pending.topic].appendleft(pending)

    def commit(self) -> None:
        pass

    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        self.produced.append((topic, value))

    def flush(self, timeout: float = 10.0) -> None:
        pass

    def close(self) -> None:
        self.closed = True
