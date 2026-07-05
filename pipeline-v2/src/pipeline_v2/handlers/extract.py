"""Extract stage handler: PDF path (in ``message.payload``) -> chunks artifact.

Manifest-driven enqueue (resolving a doc_id to a PDF path via the migrated
manifest) comes later; for now the extract stage message simply carries
``payload["pdf_path"]`` — the absolute or relative path of the PDF to
extract — plus optional ``payload["title"]`` / ``payload["source_url"]``
used only as chunk metadata.

Extract path: reuse v1's cached docling chunks, else PyMuPDF
------------------------------------------------------------
For the SOTA rebuild this handler is **cache-first**: it looks up the doc's
PDF-sha in the corpus's v1 ``extract_cache`` (docling, default profile) and, on
a hit (~97-99% of docs), reuses those chunks verbatim — giving v1-identical
``chunk_index`` + ``canonical_sha256`` (so the RLR contextual cache hits) and
the full-text docling structure the research found load-bearing, with NO docling
runtime dependency in this CPU-only stage. On a cache miss it falls back to
PyMuPDF (below). graph-layout-rag offers two PDF backends
(``graph_layout_rag.pdf_text`` / PyMuPDF, and ``graph_layout_rag.docling_text``
/ Docling, gated behind the ``docling`` extra); the PyMuPDF fallback is used
because:

* It's graph-layout-rag's own default (``default_pdf_backend()`` returns
  ``"pymupdf"`` unless ``GRAPH_RAG_PDF_BACKEND=docling`` is set) — reaching
  for the non-default backend needs a specific reason, not just availability.
* Docling pulls its own large model stack (layout/table/OCR models) and, per
  ``docling_text.extract_pages_docling``, spawns a SIGALRM-guarded subprocess
  path with GPU-device env knobs (``GRAPH_RAG_DOCLING_DEVICE`` etc.) — all of
  that cuts directly against this stage's CPU-only, no-CUDA requirement and
  would roughly double pipeline-v2's already-heavy transitive dependency
  closure (graph-layout-rag -> rag-common -> torch/sentence-transformers)
  for a backend this stage has no use for.
* PyMuPDF is synchronous, pure C-extension (no model weights, no GPU), and
  fast enough for the message-at-a-time throughput this stage handler needs.

Chunking reuses graph-layout-rag's ``ingest.chunk.chunk_pages`` unchanged
(paragraph/heading/list/table block assembly over whatever Markdown-ish
structure PyMuPDF's plain text happens to contain — headings simply won't be
detected without a Markdown-emitting backend, which is fine for this stage;
retrieval-quality chunking variants are a v1/E1 concern, not P2's).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pipeline_v2.artifacts import ArtifactStore, artifact_store_from_env
from pipeline_v2.consumers import HandlerResult
from pipeline_v2.contracts import StageMessage

logger = logging.getLogger(__name__)

# PDF extraction backend is a deliberate, swappable choice — set per run via the
# PIPELINE_V2_PDF_BACKEND env (same container, different flag). "docling" gives
# the full-text markdown structure the research found load-bearing; "pymupdf" is
# the fast plain-text fallback. The chosen backend also selects which
# extract_cache namespace we read/write, so a docling run reuses v1's docling
# cache and (re)populates it for the uncached tail.
DEFAULT_PDF_BACKEND = "pymupdf"
_VALID_PDF_BACKENDS = ("docling", "pymupdf")


def _pdf_backend() -> str:
    raw = os.getenv("PIPELINE_V2_PDF_BACKEND", DEFAULT_PDF_BACKEND).strip().lower()
    return raw if raw in _VALID_PDF_BACKENDS else DEFAULT_PDF_BACKEND


class ExtractCudaGuardError(RuntimeError):
    """Raised when a CUDA device is visible to a process that must be CPU-only."""


def _assert_cpu_only() -> None:
    """Guard: this stage must never touch a GPU. No-op if torch isn't importable."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        raise ExtractCudaGuardError(
            "pipeline-v2 extract stage is CPU-only, but a CUDA device is visible "
            "to this process; refusing to run"
        )


def make_extract_handler(store: ArtifactStore | None = None):
    """Build the extract stage handler.

    ``store`` defaults to a :class:`LocalArtifactStore` rooted at the env-
    configured (or default) pipeline-v2 data dir. Injectable for tests.
    """
    artifact_store = store if store is not None else artifact_store_from_env()

    def _handler(message: StageMessage) -> HandlerResult:
        try:
            return _extract_one(message, artifact_store)
        except Exception as exc:  # noqa: BLE001 - never raise through the consumer loop
            logger.warning(
                "extract handler degraded doc_id=%s corpus=%s: %s",
                message.doc_id,
                message.corpus,
                exc,
            )
            return HandlerResult(state="degraded", reason=f"unexpected extract error: {exc}")

    return _handler


# Corpus → the v1 tool package whose ingest modules produced the docling
# extract_cache + (for RLR) the contextual_cache. Reusing the RIGHT tool's
# cache + chunker is what makes v2 chunk_index / canonical_sha256 match v1 —
# the precondition for both caches to hit (see docs/…decision log, Phase 0a).
_CORPUS_PKG = {"glr": "graph_layout_rag", "rlr": "rag_literature_rag"}


def _corpus_modules(corpus: str):
    """Import the per-corpus v1 ingest modules (GLR vs RLR use different chunkers).

    Using the RIGHT tool's modules is what makes v2 chunk_index/canonical_sha256
    match v1 (the precondition for the extract + contextual caches to line up).
    """
    import importlib

    pkg = _CORPUS_PKG[corpus]
    return {
        "extract_cache": importlib.import_module(f"{pkg}.ingest.extract_cache"),
        "chunk": importlib.import_module(f"{pkg}.ingest.chunk"),
        "extract": importlib.import_module(f"{pkg}.ingest.extract"),
        "manifest": importlib.import_module(f"{pkg}.manifest"),
        "pdf_text": importlib.import_module(f"{pkg}.pdf_text"),
        "docling_text": importlib.import_module(f"{pkg}.docling_text"),
    }


def _records_from_chunks(chunks, embed_input_text) -> list[dict]:
    return [
        {
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "title": c.title,
            "text": c.text,
            "embed_text": embed_input_text(c),
            "page": c.page,
            "page_end": c.page_end,
            "section_path": c.section_path,
            "source_url": c.source_url,
        }
        for c in chunks
    ]


def _extract_one(message: StageMessage, store: ArtifactStore) -> HandlerResult:
    _assert_cpu_only()

    pdf_path_raw = message.payload.get("pdf_path")
    if not pdf_path_raw:
        return HandlerResult(state="degraded", reason="payload missing 'pdf_path'")

    pdf_path = Path(pdf_path_raw)
    if not pdf_path.is_file():
        return HandlerResult(state="degraded", reason=f"pdf_path does not exist: {pdf_path}")

    if message.corpus not in _CORPUS_PKG:
        return HandlerResult(state="degraded", reason=f"unknown corpus: {message.corpus}")

    backend = _pdf_backend()
    mods = _corpus_modules(message.corpus)
    extract_cache = mods["extract_cache"]
    chunk_pages = mods["chunk"].chunk_pages
    embed_input_text = mods["chunk"].embed_input_text
    chunking_fingerprint = mods["chunk"].chunking_fingerprint
    sha = message.content_sha256

    # 1. Cache-first for the CHOSEN backend: a docling run reuses v1's docling
    #    chunks (byte-identical chunk_index → contextual cache hits); a pymupdf
    #    run reuses pymupdf chunks. Cheap and gives v1 fidelity for free.
    try:
        cached = extract_cache.get(sha, backend, None) if sha else None
    except Exception as exc:  # noqa: BLE001 — cache read is best-effort
        logger.debug("extract cache read failed (%s/%s): %s", message.corpus, backend, exc)
        cached = None
    if cached:
        return _emit_chunks(
            message,
            store,
            _records_from_chunks(cached, embed_input_text),
            pdf_backend=backend,
            chunking_fingerprint=chunking_fingerprint(),
        )

    # 2. Cache miss → extract with the chosen backend. docling gives the full-text
    #    markdown structure the research found load-bearing (heavier, CPU); pymupdf
    #    is the fast plain-text path.
    if backend == "docling":
        result = mods["docling_text"].extract_pages_docling(pdf_path, clean=True)
    else:
        result = mods["pdf_text"].extract_pages_from_path(pdf_path, clean=True)
    if not result.ok:
        reason = result.open_error or "no extractable text (empty/corrupt PDF)"
        return HandlerResult(state="degraded", reason=f"{backend} extraction failed: {reason}")

    PageText = mods["extract"].PageText
    pages = [PageText(page=page_no, text=text) for page_no, text in result.pages]
    item = mods["manifest"].ManifestItem(
        id=message.doc_id,
        title=str(message.payload.get("title") or message.doc_id),
        source="pipeline-v2",
        url=str(message.payload.get("source_url") or ""),
        status="ok",
        sha256=sha,
    )
    chunks = chunk_pages(item, pages, canonical_sha256=sha)
    if not chunks:
        return HandlerResult(
            state="degraded", reason=f"{backend} extracted but produced zero chunks"
        )

    # 3. Populate the extract_cache so the (expensive) docling tail is reused on
    #    any re-run — mirrors v1 ingest/run._extract_pdf_task.
    if sha:
        try:
            extract_cache.put(sha, backend, chunks, None)
        except Exception as exc:  # noqa: BLE001 — caching is best-effort
            logger.debug("extract cache write failed (%s/%s): %s", message.corpus, backend, exc)

    return _emit_chunks(
        message,
        store,
        _records_from_chunks(chunks, embed_input_text),
        pdf_backend=backend,
        chunking_fingerprint=chunking_fingerprint(),
    )


def _emit_chunks(
    message: StageMessage,
    store: ArtifactStore,
    chunk_records: list[dict],
    *,
    pdf_backend: str,
    chunking_fingerprint: object,
) -> HandlerResult:
    """Write the chunks artifact + return the ok result (shared by cache-reuse and PyMuPDF paths)."""
    if not chunk_records:
        return HandlerResult(state="degraded", reason="no chunks produced")
    artifact_body = {
        "doc_id": message.doc_id,
        "corpus": message.corpus,
        "content_sha256": message.content_sha256,
        "stage_fingerprint": message.stage_fingerprint,
        "pdf_backend": pdf_backend,
        "chunking_fingerprint": chunking_fingerprint,
        "chunks": chunk_records,
    }
    content = json.dumps(artifact_body, sort_keys=True).encode("utf-8")
    artifact_uri = store.put(
        content,
        key_parts=["extract", message.corpus, message.doc_id, message.stage_fingerprint],
        suffix=".json",
    )
    return HandlerResult(
        state="ok",
        artifact_uri=artifact_uri,
        output_payload={
            "chunks_artifact_uri": artifact_uri,
            "chunk_count": len(chunk_records),
        },
    )
