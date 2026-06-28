"""TLDR-prefix chunk augmentation (T9 A/B arm).

Prepends each paper's one-line TLDR (Semantic Scholar `tldr.text`, stored in the
``papers_meta`` table of ``citations.sqlite`` during metadata enrichment) to the
chunk's *embed/BM25* text. The stored display text is left clean. Gated on the embed
profile name so production indexes are never touched — only profiles whose name
contains ``tldr`` are augmented, e.g. ``cuda-qwen0.6b-tldr-v1``.

Unlike contextual augmentation, the TLDR is already-computed metadata: there is no LLM
call, just a batched lookup keyed by ``doc_id``. Fails soft — a chunk whose doc has no
TLDR (or when the store is absent) is left unaugmented rather than aborting ingest.

This re-tests the prior NULL prefix-embedding results (section-v1, contextual-v1 both
gate-FAILED on this BM25-dominant corpus) with a *document-level summary* prefix instead
of a section path or an LLM context line.
"""
from __future__ import annotations

import logging

from graph_layout_rag.ingest.chunk import TextChunk

log = logging.getLogger("graph_layout_rag.ingest.tldr_prefix")


def is_tldr_profile(profile: str | None) -> bool:
    return bool(profile) and "tldr" in profile.lower()


def _tldr_by_doc(doc_ids: set[str]) -> dict[str, str]:
    """Batch-load non-empty TLDRs for the given doc_ids. Empty dict if no store."""
    from graph_layout_rag.citation_store import connect, paper_meta_for_doc
    from graph_layout_rag.paths import CITATIONS_DB_PATH

    if not CITATIONS_DB_PATH.exists():
        return {}
    out: dict[str, str] = {}
    db = connect()
    try:
        for doc_id in doc_ids:
            meta = paper_meta_for_doc(db, doc_id)
            tldr = (meta.tldr or "").strip() if meta else ""
            if tldr:
                out[doc_id] = tldr
    finally:
        db.close()
    return out


def augment_texts_for_tldr(chunks: list[TextChunk], texts: list[str]) -> list[str]:
    """Return embed/BM25 texts with a ``TLDR: <summary>`` line prepended per chunk.

    All chunks of a doc share the doc's TLDR (it is a paper-level summary). Chunks whose
    doc lacks a TLDR are left unchanged, so coverage is partial-by-design and the arm is a
    clean superset-prefix A/B vs the base profile.
    """
    tldrs = _tldr_by_doc({c.doc_id for c in chunks})
    if not tldrs:
        log.warning("tldr augmentation: no TLDRs found for %d chunk(s); texts unchanged", len(chunks))
        return texts
    augmented: list[str] = []
    hits = 0
    for chunk, text in zip(chunks, texts):
        tldr = tldrs.get(chunk.doc_id)
        if tldr:
            augmented.append(f"TLDR: {tldr}\n{text}")
            hits += 1
        else:
            augmented.append(text)
    log.info(
        "tldr augmentation: prefixed %d/%d chunk(s) across %d doc(s) with TLDRs",
        hits, len(chunks), len(tldrs),
    )
    return augmented
