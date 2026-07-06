"""App-level dead-letter queue.

Kafka share groups have NO automatic DLQ (spike REPORT.md Milestone 5): a
record that exhausts its delivery attempts is silently archived by the
broker — no topic, no event, nothing to alert on. This module is what
makes exhausted/poison messages observable and recoverable:

* a record whose ``delivery_count`` has reached :data:`DEFAULT_THRESHOLD`
  (default 4 — one below the broker's own
  ``group.share.delivery.count.limit`` default of 5, so we act before the
  broker would have archived it anyway) is produced to ``deadletter.<stage>``
  with reason + attempt metadata, the original message is ACCEPTed (a
  deliberate archive, not a redelivery-eligible release), and a ``'failed'``
  outcome is recorded in ``work_state`` via the existing repository.
* poison messages (failed contract validation) go through the same path
  immediately, regardless of delivery count — see
  :mod:`pipeline_v2.consumers`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pipeline_v2.contracts import DeadLetterMessage, StageMessage, deadletter_topic
from pipeline_v2.messaging import WorkQueue
from pipeline_v2.workstate import WorkStateRepository

logger = logging.getLogger(__name__)

# One below the broker's default group.share.delivery.count.limit (5), so
# the app-level DLQ fires before KIP-932 would silently archive the record
# out from under us.
DEFAULT_THRESHOLD = 4


@dataclass
class DeadLetterProducer:
    """Sends poisoned/exhausted messages to their stage's DLQ topic and
    records the outcome in work_state."""

    queue: WorkQueue
    repo: WorkStateRepository
    threshold: int = DEFAULT_THRESHOLD

    def should_deadletter(self, delivery_count: int) -> bool:
        return delivery_count >= self.threshold

    def send(
        self,
        *,
        stage: str,
        original_topic: str,
        reason: str,
        delivery_count: int,
        message: StageMessage | None = None,
        identity: dict[str, str] | None = None,
        run_id: str | None = None,
    ) -> None:
        """Produce to ``deadletter.<stage>`` and record a 'failed' work_state row.

        ``message`` is the validated contract instance when available
        (delivery-count exhaustion of an otherwise-valid message). For
        poison messages that failed validation, pass ``identity`` — the
        best-effort partial parse from
        :func:`pipeline_v2.contracts.try_extract_identity` — instead;
        whatever of doc_id/corpus/content_sha256/stage_fingerprint could be
        recovered is used, with ``"unknown"`` filling any gaps, so the DLQ
        message and work_state row are always written even for garbage
        payloads.
        """
        topic = deadletter_topic(stage)
        identity = identity or {}
        doc_id = (message.doc_id if message is not None else identity.get("doc_id")) or "unknown"
        corpus = (message.corpus if message is not None else identity.get("corpus")) or "unknown"
        content_sha256 = (
            message.content_sha256 if message is not None else identity.get("content_sha256")
        ) or "unknown"
        stage_fingerprint = (
            message.stage_fingerprint if message is not None else identity.get("stage_fingerprint")
        ) or "unknown"
        attempt = message.attempt if message is not None else delivery_count
        payload = message.payload if message is not None else {}

        dlq_message = DeadLetterMessage(
            doc_id=doc_id,
            corpus=corpus,
            content_sha256=content_sha256,
            stage_fingerprint=stage_fingerprint,
            attempt=attempt,
            payload=payload,
            reason=reason,
            original_topic=original_topic,
            stage=stage,
        )
        self.queue.produce(topic, dlq_message.model_dump_json().encode("utf-8"))
        self.queue.flush(timeout=10.0)

        self.repo.upsert_outcome(
            doc_id=doc_id,
            corpus=corpus,
            stage=stage,
            content_sha256=content_sha256,
            stage_fingerprint=stage_fingerprint,
            state="failed",
            reason=reason,
            run_id=run_id,
        )
        logger.warning(
            "deadlettered doc_id=%s corpus=%s stage=%s delivery_count=%d reason=%s",
            doc_id,
            corpus,
            stage,
            delivery_count,
            reason,
        )
