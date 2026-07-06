"""End-to-end P2 integration test: extract -> embed -> indexer over real Kafka + Postgres.

Three documents flow through the whole real pipeline in one pass:

* a good, extractable PDF -> should land in both the v2 LanceDB table and
  the v2 tantivy index;
* an empty/corrupt "PDF" -> extract reports it ``degraded`` and it never
  reaches embed/indexer (the consumer loop only propagates ``state=='ok'``
  downstream), so it must be provably absent from both indexes;
* a poison (invalid JSON) message on the extract stage's input topic ->
  goes straight to ``deadletter.extract``, no handler ever runs.

A second pass re-produces the same two "real" messages (good + degraded) and
re-runs every stage to prove idempotency: no duplicate rows land in the
Lance table for the good doc.

Every consumer loop below is bounded (`max_iterations`, short `poll_timeout`)
per the anti-stall discipline used throughout this suite (see
`test_messaging_integration.py`'s module docstring) — nothing here can hang.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from pipeline_v2 import contracts as contracts_module
from pipeline_v2 import deadletter as deadletter_module
from pipeline_v2.artifacts import LocalArtifactStore
from pipeline_v2.consumers import StageConfig, StageConsumer
from pipeline_v2.contracts import (
    ChunksEmbeddedMessage,
    ChunksReadyMessage,
    DeadLetterMessage,
    DocsEnrichedMessage,
    IndexCommittedMessage,
)
from pipeline_v2.deadletter import DeadLetterProducer
from pipeline_v2.handlers.embed import make_embed_handler
from pipeline_v2.handlers.extract import make_extract_handler
from pipeline_v2.handlers.indexer import (
    V2IndexPaths,
    bm25_doc_ids,
    lance_doc_chunk_ids,
    make_index_writer,
)
from pipeline_v2.messaging import ShareGroupQueue
from pipeline_v2.workstate import PostgresWorkStateRepository
from test_handler_embed import _StubGatewayServer

pytestmark = pytest.mark.integration

CORPUS = "e2e-corpus"


@pytest.fixture
def pipeline_topics(kafka_bootstrap: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    from confluent_kafka.admin import AdminClient, NewTopic

    suffix = uuid.uuid4().hex[:8]
    names = {
        "docs.enriched": f"docs.enriched.{suffix}",
        "chunks.ready": f"chunks.ready.{suffix}",
        "chunks.embedded": f"chunks.embedded.{suffix}",
        "index.committed": f"index.committed.{suffix}",
        "deadletter.extract": f"deadletter.extract.{suffix}",
        "deadletter.embed": f"deadletter.embed.{suffix}",
        "deadletter.indexer": f"deadletter.indexer.{suffix}",
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

    monkeypatch.setitem(contracts_module.TOPIC_MODELS, names["docs.enriched"], DocsEnrichedMessage)
    monkeypatch.setitem(contracts_module.TOPIC_MODELS, names["chunks.ready"], ChunksReadyMessage)
    monkeypatch.setitem(
        contracts_module.TOPIC_MODELS, names["chunks.embedded"], ChunksEmbeddedMessage
    )
    monkeypatch.setitem(
        contracts_module.TOPIC_MODELS, names["index.committed"], IndexCommittedMessage
    )
    for stage in ("extract", "embed", "indexer"):
        monkeypatch.setitem(
            contracts_module.TOPIC_MODELS, names[f"deadletter.{stage}"], DeadLetterMessage
        )
    monkeypatch.setattr(
        deadletter_module, "deadletter_topic", lambda stage: names[f"deadletter.{stage}"]
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


def _make_pdf(path: Path, text: str) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _stage_message(doc_id: str, *, payload: dict) -> bytes:
    return json.dumps(
        {
            "doc_id": doc_id,
            "corpus": CORPUS,
            "content_sha256": f"sha-{doc_id}",
            "stage_fingerprint": "fp-e2e-v1",
            "attempt": 1,
            "payload": payload,
        }
    ).encode()


def _drain_topic(bootstrap: str, topic: str, group: str, *, max_iterations: int = 8) -> list[dict]:
    from pipeline_v2.messaging import AckAction

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
    finally:
        queue.close()
    return out


def _run_stage(
    *,
    kafka_bootstrap: str,
    repo: PostgresWorkStateRepository,
    name: str,
    input_topic: str,
    output_topic: str | None,
    handler,
    group_prefix: str,
    max_iterations: int = 10,
) -> None:
    config = StageConfig(
        name=name, input_topic=input_topic, handler=handler, output_topic=output_topic
    )
    group = f"{group_prefix}-{uuid.uuid4().hex[:8]}"
    queue = ShareGroupQueue(bootstrap_servers=kafka_bootstrap, group_id=group, topics=[input_topic])
    dlq = DeadLetterProducer(queue=queue, repo=repo)
    consumer = StageConsumer(queue=queue, repo=repo, config=config, dlq=dlq, run_id=group)
    try:
        consumer.run_forever(max_iterations=max_iterations, poll_timeout=2.0)
    finally:
        queue.close()


def test_extract_embed_indexer_e2e_with_degraded_and_poison(
    kafka_bootstrap, pipeline_topics, pg_repo, tmp_path
):
    good_pdf = tmp_path / "good.pdf"
    _make_pdf(good_pdf, "Hello pipeline-v2 end to end test document.")
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"this is not a pdf")

    # Seed: one good doc, one degraded (corrupt PDF), one poison message.
    _produce_raw(
        kafka_bootstrap,
        pipeline_topics["docs.enriched"],
        _stage_message("good-doc", payload={"pdf_path": str(good_pdf), "title": "Good Doc"}),
    )
    _produce_raw(
        kafka_bootstrap,
        pipeline_topics["docs.enriched"],
        _stage_message("bad-doc", payload={"pdf_path": str(corrupt_pdf), "title": "Bad Doc"}),
    )
    _produce_raw(kafka_bootstrap, pipeline_topics["docs.enriched"], b"not valid json at all {{{")

    store = LocalArtifactStore(root=tmp_path / "artifacts")
    indexes_root = tmp_path / "indexes"
    gateway = _StubGatewayServer([200, 200, 200, 200])  # generous: good run + rerun

    try:
        extract_handler = make_extract_handler(store=store)
        embed_handler = make_embed_handler(store=store, gateway_url=gateway.url, max_retries=3)
        index_writer = make_index_writer(store=store, indexes_root=indexes_root)

        def _run_all_stages():
            _run_stage(
                kafka_bootstrap=kafka_bootstrap,
                repo=pg_repo,
                name="extract",
                input_topic=pipeline_topics["docs.enriched"],
                output_topic=pipeline_topics["chunks.ready"],
                handler=extract_handler,
                group_prefix="e2e-extract",
            )
            _run_stage(
                kafka_bootstrap=kafka_bootstrap,
                repo=pg_repo,
                name="embed",
                input_topic=pipeline_topics["chunks.ready"],
                output_topic=pipeline_topics["chunks.embedded"],
                handler=embed_handler,
                group_prefix="e2e-embed",
            )
            from pipeline_v2.consumers import make_indexer_handler

            indexer_handler = make_indexer_handler(
                pg_repo, gate_stage="embed", index_writer=index_writer
            )
            _run_stage(
                kafka_bootstrap=kafka_bootstrap,
                repo=pg_repo,
                name="indexer",
                input_topic=pipeline_topics["chunks.embedded"],
                output_topic=pipeline_topics["index.committed"],
                handler=indexer_handler,
                group_prefix="e2e-indexer",
            )

        # --- Pass 1 -------------------------------------------------------
        _run_all_stages()

        extract_good = pg_repo.get("good-doc", CORPUS, "extract")
        assert extract_good is not None and extract_good.state == "ok"
        embed_good = pg_repo.get("good-doc", CORPUS, "embed")
        assert embed_good is not None and embed_good.state == "ok"
        indexer_good = pg_repo.get("good-doc", CORPUS, "indexer")
        assert indexer_good is not None and indexer_good.state == "ok"

        extract_bad = pg_repo.get("bad-doc", CORPUS, "extract")
        assert extract_bad is not None and extract_bad.state == "degraded"
        # A degraded extract never produces a chunks.ready message, so the bad
        # doc must never even get an embed/indexer work_state row.
        assert pg_repo.get("bad-doc", CORPUS, "embed") is None
        assert pg_repo.get("bad-doc", CORPUS, "indexer") is None

        dlq_messages = _drain_topic(
            kafka_bootstrap,
            pipeline_topics["deadletter.extract"],
            f"e2e-poison-check-{uuid.uuid4().hex[:8]}",
        )
        assert len(dlq_messages) == 1
        assert "poison" in dlq_messages[0]["reason"]

        paths = V2IndexPaths.for_corpus(CORPUS, root=indexes_root)
        lance_ids = lance_doc_chunk_ids(paths.lance_dir, corpus=CORPUS)
        assert lance_ids, "good doc's chunks must be in the v2 Lance table"
        assert all(cid.startswith("good-doc:") for cid in lance_ids)
        assert bm25_doc_ids(paths.bm25_dir) == {"good-doc"}

        import lancedb

        rows_after_pass1 = lancedb.connect(str(paths.lance_dir)).open_table("chunks").count_rows()

        # --- Pass 2: rerun the exact same input messages (idempotency) ----
        _produce_raw(
            kafka_bootstrap,
            pipeline_topics["docs.enriched"],
            _stage_message("good-doc", payload={"pdf_path": str(good_pdf), "title": "Good Doc"}),
        )
        _produce_raw(
            kafka_bootstrap,
            pipeline_topics["docs.enriched"],
            _stage_message("bad-doc", payload={"pdf_path": str(corrupt_pdf), "title": "Bad Doc"}),
        )
        _run_all_stages()

        rows_after_pass2 = lancedb.connect(str(paths.lance_dir)).open_table("chunks").count_rows()
        assert rows_after_pass2 == rows_after_pass1, "rerun must not duplicate index rows"
        assert lance_doc_chunk_ids(paths.lance_dir, corpus=CORPUS) == lance_ids
        assert bm25_doc_ids(paths.bm25_dir) == {"good-doc"}
    finally:
        gateway.close()
