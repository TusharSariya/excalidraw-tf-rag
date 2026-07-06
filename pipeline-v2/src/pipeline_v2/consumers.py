"""Generic per-stage consumer loop, plus the three thin stage configs.

Loop shape (one iteration of :meth:`StageConsumer.run_once`, per polled
message):

1. **poll** — one record off the queue.
2. **validate contract** — :func:`pipeline_v2.contracts.validate_message`
   against the input topic's registered model. Invalid = poison: produced
   straight to ``deadletter.<stage>`` and ACCEPTed, no work_state identity
   required (best-effort partial parse only).
3. **idempotency check** — if ``work_state`` already has a row for
   ``(doc_id, corpus, stage)`` with ``state='ok'`` and the *same*
   ``content_sha256``+``stage_fingerprint`` as this message, the stage has
   already produced this exact outcome: ACCEPT and skip (no handler call,
   no duplicate next-stage message, no work_state churn — mirrors
   ``WorkStateRepository.upsert_outcome``'s own no-op rule).
4. **handler** — the injected callable does the actual stage work and
   returns a :class:`HandlerResult`. Handlers are pure business logic; they
   never touch the queue or work_state directly.
5. **record outcome** — the result's state (ok/degraded/failed) is
   upserted into work_state via the existing repository.
6. **produce next-stage message** — ONLY on ``state == 'ok'`` (degraded and
   failed outcomes never propagate downstream — this is the same
   "only 'ok' is indexable" discipline as ``indexable_work_state``, applied
   at the messaging layer too).
7. **ack** — ACCEPT on ok/degraded (both are "we handled this record", just
   with a different work_state outcome); on ``failed`` (either an explicit
   HandlerResult or an uncaught handler exception), either DLQ+ACCEPT (if
   :meth:`DeadLetterProducer.should_deadletter` says the delivery count is
   exhausted) or RELEASE for redelivery.

Handlers are injected callables (``Handler``) so extract/embed/indexer stay
swappable without touching the loop. ``extract``/``embed``/``indexer``'s
``*_stage_config`` builders default to the real handlers in
:mod:`pipeline_v2.handlers` (extract/chunk a PDF, call the gpu-gateway,
write the v2 LanceDB+tantivy index) but accept an explicit ``handler``/
``index_writer`` override so tests can keep injecting fakes without
touching this module. ``indexer`` additionally enforces the structural gate
described in :func:`make_indexer_handler`'s docstring ahead of the injected
index-write body.

Topic flow: ``docs.enriched`` (seeded upstream — for now, directly with
``payload["pdf_path"]``; manifest-driven enqueue comes later) -> **extract**
(extract text + chunk) -> ``chunks.ready`` -> **embed** (call gpu-gateway) ->
``chunks.embedded`` -> **indexer** (write index) -> ``index.committed``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pipeline_v2.contracts import (
    ContractValidationError,
    StageMessage,
    try_extract_identity,
    validate_message,
)
from pipeline_v2.deadletter import DeadLetterProducer
from pipeline_v2.messaging import AckAction, QueueMessage, WorkQueue
from pipeline_v2.metrics import (
    HANDLER_SECONDS,
    MESSAGES_ACKED_TOTAL,
    MESSAGES_DLQ_TOTAL,
    MESSAGES_PROCESSED_TOTAL,
    MESSAGES_RELEASED_TOTAL,
)
from pipeline_v2.workstate import State, WorkStateRepository

logger = logging.getLogger(__name__)


@dataclass
class HandlerResult:
    """What a stage handler reports back to the consumer loop."""

    state: State
    reason: str | None = None
    artifact_uri: str | None = None
    # Payload merged into the outgoing next-stage message when state=='ok'
    # and output_topic is configured. Ignored otherwise.
    output_payload: dict = field(default_factory=dict)


Handler = Callable[[StageMessage], HandlerResult]


class IndexerGateError(Exception):
    """Raised by the indexer's gate check when the upstream stage isn't 'ok' yet."""


def make_indexer_handler(
    repo: WorkStateRepository,
    gate_stage: str = "embed",
    index_writer: Handler | None = None,
) -> Handler:
    """Build the indexer handler: structural gate first, then the index-write body.

    The gate is the requirement from the messaging-backbone spec: the
    indexer stage REFUSES any message whose work_state row for the upstream
    stage (``embed`` by default) isn't ``state == 'ok'``. This is defense
    in depth on top of ``indexable_work_state`` (the DB-level "only read
    'ok' rows" guard) — even if a stray/duplicate/out-of-order message
    reaches the indexer's input topic, the indexer itself refuses to act on
    it unless work_state independently confirms the upstream stage
    succeeded for this exact doc_id/corpus/sha/fingerprint.

    Raises :class:`IndexerGateError` (mapped to a 'failed' outcome by the
    consumer loop, same as any other handler exception) when the gate
    fails. When the gate passes, delegates to ``index_writer`` (defaults to
    :func:`pipeline_v2.handlers.indexer.make_index_writer`'s real
    LanceDB+tantivy writer, built lazily so importing this module never
    pulls in lancedb/tantivy unless the indexer stage is actually built);
    pass an explicit ``index_writer`` to keep injecting a fake in tests.
    """
    writer = index_writer
    if writer is None:
        from pipeline_v2.handlers.indexer import make_index_writer

        writer = make_index_writer()

    def _indexer_handler(message: StageMessage) -> HandlerResult:
        row = repo.get(message.doc_id, message.corpus, gate_stage)
        if row is None:
            raise IndexerGateError(
                f"no {gate_stage!r} work_state row for doc_id={message.doc_id!r} "
                f"corpus={message.corpus!r}; refusing to index"
            )
        if row.state != "ok":
            raise IndexerGateError(
                f"{gate_stage!r} work_state row for doc_id={message.doc_id!r} "
                f"corpus={message.corpus!r} is state={row.state!r}, not 'ok'; refusing to index"
            )
        if row.content_sha256 != message.content_sha256 or row.stage_fingerprint != (
            message.stage_fingerprint
        ):
            raise IndexerGateError(
                f"{gate_stage!r} work_state row for doc_id={message.doc_id!r} "
                f"corpus={message.corpus!r} does not match this message's "
                f"content_sha256/stage_fingerprint; refusing to index (stale message)"
            )
        return writer(message)

    return _indexer_handler


@dataclass
class StageConfig:
    """Wiring for one pipeline stage: which topics, which handler."""

    name: str
    input_topic: str
    handler: Handler
    output_topic: str | None = None


def extract_stage_config(
    handler: Handler | None = None, output_topic: str = "chunks.ready"
) -> StageConfig:
    """Extract stage: text-extract + chunk a PDF, write a chunks artifact.

    Defaults to the real handler (:func:`pipeline_v2.handlers.extract.make_extract_handler`),
    imported lazily so this module stays importable without graph-layout-rag
    pulled in unless the extract stage is actually built. Pass ``handler``
    to inject a fake in tests.
    """
    if handler is None:
        from pipeline_v2.handlers.extract import make_extract_handler

        handler = make_extract_handler()
    return StageConfig(
        name="extract", input_topic="docs.enriched", handler=handler, output_topic=output_topic
    )


def embed_stage_config(
    handler: Handler | None = None, output_topic: str = "chunks.embedded"
) -> StageConfig:
    """Embed stage: call the gpu-gateway ``/v1/embeddings``, write an embeddings artifact.

    Defaults to the real handler (:func:`pipeline_v2.handlers.embed.make_embed_handler`),
    imported lazily. Pass ``handler`` to inject a fake in tests.
    """
    if handler is None:
        from pipeline_v2.handlers.embed import make_embed_handler

        handler = make_embed_handler()
    return StageConfig(
        name="embed",
        input_topic="chunks.ready",
        handler=handler,
        output_topic=output_topic,
    )


def indexer_stage_config(
    repo: WorkStateRepository, index_writer: Handler | None = None
) -> StageConfig:
    return StageConfig(
        name="indexer",
        input_topic="chunks.embedded",
        handler=make_indexer_handler(repo, index_writer=index_writer),
        output_topic="index.committed",
    )


STAGE_CONFIG_BUILDERS: dict[str, Callable[..., StageConfig]] = {
    "extract": lambda repo: extract_stage_config(),
    "embed": lambda repo: embed_stage_config(),
    "indexer": indexer_stage_config,
}


class StageConsumer:
    """Runs :class:`StageConfig` against a :class:`WorkQueue` + work_state repo."""

    def __init__(
        self,
        *,
        queue: WorkQueue,
        repo: WorkStateRepository,
        config: StageConfig,
        dlq: DeadLetterProducer,
        run_id: str | None = None,
    ) -> None:
        self.queue = queue
        self.repo = repo
        self.config = config
        self.dlq = dlq
        self.run_id = run_id

    def run_once(self, timeout: float = 1.0) -> int:
        """Poll one batch and process every record in it. Returns records processed."""
        batch = self.queue.poll_batch(timeout=timeout)
        for qmsg in batch:
            self._process_one(qmsg)
        if batch:
            self.queue.commit()
        return len(batch)

    def run_forever(self, *, max_iterations: int | None = None, poll_timeout: float = 1.0) -> int:
        """Bounded-by-default loop: pass max_iterations in tests/CI to avoid an unbounded poll."""
        iterations = 0
        total = 0
        while max_iterations is None or iterations < max_iterations:
            total += self.run_once(timeout=poll_timeout)
            iterations += 1
        return total

    def _process_one(self, qmsg: QueueMessage) -> None:
        try:
            message = validate_message(qmsg.topic, qmsg.value)
        except ContractValidationError as exc:
            identity = try_extract_identity(qmsg.value)
            poison_corpus = identity.get("corpus") if identity else None
            self.dlq.send(
                stage=self.config.name,
                original_topic=qmsg.topic,
                reason=f"poison message: {exc}",
                delivery_count=qmsg.delivery_count,
                identity=identity,
                run_id=self.run_id,
            )
            MESSAGES_DLQ_TOTAL.labels(
                stage=self.config.name, corpus=poison_corpus or "unknown"
            ).inc()
            self.queue.ack(qmsg, AckAction.ACCEPT)
            MESSAGES_ACKED_TOTAL.labels(
                stage=self.config.name, corpus=poison_corpus or "unknown"
            ).inc()
            return

        existing = self.repo.get(message.doc_id, message.corpus, self.config.name)
        if (
            existing is not None
            and existing.state == "ok"
            and existing.content_sha256 == message.content_sha256
            and existing.stage_fingerprint == message.stage_fingerprint
        ):
            logger.info(
                "idempotent skip doc_id=%s corpus=%s stage=%s (already ok)",
                message.doc_id,
                message.corpus,
                self.config.name,
            )
            self.queue.ack(qmsg, AckAction.ACCEPT)
            MESSAGES_ACKED_TOTAL.labels(stage=self.config.name, corpus=message.corpus).inc()
            return

        MESSAGES_PROCESSED_TOTAL.labels(stage=self.config.name, corpus=message.corpus).inc()
        start = time.perf_counter()
        try:
            result = self.config.handler(message)
        except Exception as exc:  # noqa: BLE001 - any handler failure is a 'failed' outcome
            HANDLER_SECONDS.labels(stage=self.config.name, corpus=message.corpus).observe(
                time.perf_counter() - start
            )
            self._handle_failure(qmsg, message, reason=str(exc))
            return
        HANDLER_SECONDS.labels(stage=self.config.name, corpus=message.corpus).observe(
            time.perf_counter() - start
        )

        self.repo.upsert_outcome(
            doc_id=message.doc_id,
            corpus=message.corpus,
            stage=self.config.name,
            content_sha256=message.content_sha256,
            stage_fingerprint=message.stage_fingerprint,
            state=result.state,
            reason=result.reason,
            artifact_uri=result.artifact_uri,
            run_id=self.run_id,
        )

        if result.state == "ok" and self.config.output_topic is not None:
            next_message = StageMessage(
                doc_id=message.doc_id,
                corpus=message.corpus,
                content_sha256=message.content_sha256,
                stage_fingerprint=message.stage_fingerprint,
                attempt=1,
                payload=result.output_payload,
            )
            self.queue.produce(
                self.config.output_topic, next_message.model_dump_json().encode("utf-8")
            )
            self.queue.flush(timeout=10.0)

        if result.state == "failed":
            self._maybe_deadletter_or_release(qmsg, message, reason=result.reason or "failed")
        else:
            self.queue.ack(qmsg, AckAction.ACCEPT)
            MESSAGES_ACKED_TOTAL.labels(stage=self.config.name, corpus=message.corpus).inc()

    def _handle_failure(self, qmsg: QueueMessage, message: StageMessage, *, reason: str) -> None:
        self.repo.upsert_outcome(
            doc_id=message.doc_id,
            corpus=message.corpus,
            stage=self.config.name,
            content_sha256=message.content_sha256,
            stage_fingerprint=message.stage_fingerprint,
            state="failed",
            reason=reason,
            run_id=self.run_id,
        )
        self._maybe_deadletter_or_release(qmsg, message, reason=reason)

    def _maybe_deadletter_or_release(
        self, qmsg: QueueMessage, message: StageMessage, *, reason: str
    ) -> None:
        if self.dlq.should_deadletter(qmsg.delivery_count):
            self.dlq.send(
                stage=self.config.name,
                original_topic=qmsg.topic,
                reason=reason,
                delivery_count=qmsg.delivery_count,
                message=message,
                run_id=self.run_id,
            )
            MESSAGES_DLQ_TOTAL.labels(stage=self.config.name, corpus=message.corpus).inc()
            self.queue.ack(qmsg, AckAction.ACCEPT)
            MESSAGES_ACKED_TOTAL.labels(stage=self.config.name, corpus=message.corpus).inc()
        else:
            self.queue.ack(qmsg, AckAction.RELEASE)
            MESSAGES_RELEASED_TOTAL.labels(stage=self.config.name, corpus=message.corpus).inc()


QueueBackend = Literal["share", "classic"]
