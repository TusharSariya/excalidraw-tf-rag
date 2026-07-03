from __future__ import annotations

import json

import pytest
from pipeline_v2.artifacts import LocalArtifactStore
from pipeline_v2.contracts import StageMessage
from pipeline_v2.handlers.extract import make_extract_handler


def _make_pdf(tmp_path, *, text: str = "Hello pipeline-v2. This is page one.") -> str:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return str(path)


def _message(**payload) -> StageMessage:
    return StageMessage(
        doc_id="doc-1",
        corpus="test-corpus",
        content_sha256="deadbeef",
        stage_fingerprint="fp-1",
        payload=payload,
    )


def test_extract_ok_writes_chunks_artifact(tmp_path):
    pdf_path = _make_pdf(tmp_path)
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    handler = make_extract_handler(store=store)

    result = handler(_message(pdf_path=pdf_path, title="A Test Paper"))

    assert result.state == "ok", result.reason
    assert result.artifact_uri is not None
    assert result.output_payload["chunks_artifact_uri"] == result.artifact_uri
    assert result.output_payload["chunk_count"] >= 1

    body = json.loads(store.get(result.artifact_uri))
    assert body["doc_id"] == "doc-1"
    assert body["pdf_backend"] == "pymupdf"
    assert len(body["chunks"]) == result.output_payload["chunk_count"]
    assert "pipeline-v2" in body["chunks"][0]["text"] or "Hello" in body["chunks"][0]["text"]


def test_extract_missing_pdf_path_is_degraded_not_raised(tmp_path):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    handler = make_extract_handler(store=store)

    result = handler(_message())

    assert result.state == "degraded"
    assert "pdf_path" in (result.reason or "")


def test_extract_nonexistent_pdf_is_degraded(tmp_path):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    handler = make_extract_handler(store=store)

    result = handler(_message(pdf_path=str(tmp_path / "does-not-exist.pdf")))

    assert result.state == "degraded"
    assert "does not exist" in (result.reason or "")


def test_extract_corrupt_pdf_is_degraded_never_raises(tmp_path):
    bad_path = tmp_path / "corrupt.pdf"
    bad_path.write_bytes(b"not a real pdf file at all")
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    handler = make_extract_handler(store=store)

    result = handler(_message(pdf_path=str(bad_path)))

    assert result.state == "degraded"
    assert result.reason


def test_extract_empty_pdf_zero_pages_is_degraded(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()  # blank page, no text
    doc.save(str(path))
    doc.close()

    store = LocalArtifactStore(root=tmp_path / "artifacts")
    handler = make_extract_handler(store=store)

    result = handler(_message(pdf_path=str(path)))

    assert result.state == "degraded"
