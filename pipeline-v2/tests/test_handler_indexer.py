from __future__ import annotations

import json

from pipeline_v2.artifacts import LocalArtifactStore
from pipeline_v2.contracts import StageMessage
from pipeline_v2.handlers.indexer import (
    V2IndexPaths,
    bm25_doc_ids,
    lance_doc_chunk_ids,
    make_index_writer,
)


def _embeddings_artifact(store, *, doc_id="doc-1", corpus="test-corpus", fp="fp", n=2, dims=4):
    body = {
        "doc_id": doc_id,
        "corpus": corpus,
        "content_sha256": "sha",
        "stage_fingerprint": fp,
        "chunks": [
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "title": "T",
                "text": f"chunk {i}",
                "page": 1,
                "page_end": 1,
                "section_path": "",
                "source_url": "",
                "vector": [0.1 * (i + 1)] * dims,
            }
            for i in range(n)
        ],
    }
    return store.put(
        json.dumps(body).encode(), key_parts=["embed", corpus, doc_id, fp], suffix=".json"
    )


def _message(artifact_uri, *, doc_id="doc-1", corpus="test-corpus", fp="fp") -> StageMessage:
    payload = {"embeddings_artifact_uri": artifact_uri} if artifact_uri else {}
    return StageMessage(
        doc_id=doc_id, corpus=corpus, content_sha256="sha", stage_fingerprint=fp, payload=payload
    )


def test_indexer_ok_writes_lance_and_bm25(tmp_path):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    indexes_root = tmp_path / "indexes"
    writer = make_index_writer(store=store, indexes_root=indexes_root)

    artifact_uri = _embeddings_artifact(store)
    result = writer(_message(artifact_uri))

    assert result.state == "ok", result.reason
    assert result.output_payload["indexed_chunk_count"] == 2
    assert "lance_version=" in (result.reason or "")

    paths = V2IndexPaths.for_corpus("test-corpus", root=indexes_root)
    chunk_ids = lance_doc_chunk_ids(paths.lance_dir, corpus="test-corpus")
    assert chunk_ids == {"doc-1:0", "doc-1:1"}
    assert bm25_doc_ids(paths.bm25_dir) == {"doc-1"}


def test_indexer_missing_artifact_payload_is_degraded(tmp_path):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    writer = make_index_writer(store=store, indexes_root=tmp_path / "indexes")

    result = writer(_message(None))

    assert result.state == "degraded"
    assert "embeddings_artifact_uri" in (result.reason or "")


def test_indexer_missing_vector_is_degraded(tmp_path):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    body = {
        "doc_id": "doc-1",
        "corpus": "test-corpus",
        "content_sha256": "sha",
        "stage_fingerprint": "fp",
        "chunks": [{"doc_id": "doc-1", "chunk_index": 0, "text": "x"}],  # no vector
    }
    artifact_uri = store.put(
        json.dumps(body).encode(), key_parts=["embed", "test-corpus", "doc-1", "fp"], suffix=".json"
    )
    writer = make_index_writer(store=store, indexes_root=tmp_path / "indexes")

    result = writer(_message(artifact_uri))

    assert result.state == "degraded"
    assert "vector" in (result.reason or "")


def test_indexer_rerun_is_idempotent_no_duplicate_rows(tmp_path):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    indexes_root = tmp_path / "indexes"
    writer = make_index_writer(store=store, indexes_root=indexes_root)

    artifact_uri = _embeddings_artifact(store)
    writer(_message(artifact_uri))
    writer(_message(artifact_uri))  # rerun of the exact same message

    paths = V2IndexPaths.for_corpus("test-corpus", root=indexes_root)
    chunk_ids = lance_doc_chunk_ids(paths.lance_dir, corpus="test-corpus")
    assert chunk_ids == {"doc-1:0", "doc-1:1"}  # no duplicates

    import lancedb

    db = lancedb.connect(str(paths.lance_dir))
    table = db.open_table("chunks")
    assert table.count_rows() == 2


def test_indexer_two_docs_do_not_clobber_each_other(tmp_path):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    indexes_root = tmp_path / "indexes"
    writer = make_index_writer(store=store, indexes_root=indexes_root)

    uri1 = _embeddings_artifact(store, doc_id="doc-1", fp="fp1")
    uri2 = _embeddings_artifact(store, doc_id="doc-2", fp="fp2")
    writer(_message(uri1, doc_id="doc-1", fp="fp1"))
    writer(_message(uri2, doc_id="doc-2", fp="fp2"))

    paths = V2IndexPaths.for_corpus("test-corpus", root=indexes_root)
    assert lance_doc_chunk_ids(paths.lance_dir) == {"doc-1:0", "doc-1:1", "doc-2:0", "doc-2:1"}
    assert bm25_doc_ids(paths.bm25_dir) == {"doc-1", "doc-2"}
