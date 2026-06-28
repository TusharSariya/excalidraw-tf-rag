"""Late-chunking embed path (Jina-style), gated on the embed profile name.

Mirrors ingest/contextual.py's name-gating so production profiles are never
touched: only profiles whose name contains "latechunk" use this path. The
window is embedded ONCE at token level (cross-chunk context preserved), then
each chunk's vector is a mean-pool of its token span within the window. Spans
are tracked at construction time by chunk.assign_late_chunk_windows (the
provenance audit showed post-hoc offset recovery is ~14% viable, so we never
recover — we build the window from the chunks and know each span exactly).

Wiring seam: ingest/index.py builds embed texts then calls embed_texts(...).
For a latechunk profile, instead assign windows on the doc's chunks and call
``embed_late_chunks(chunks, config)`` to get one vector per chunk. This module
provides that call; index.py wiring is intentionally left as a documented seam
(see embed_late_chunks docstring) to avoid disturbing the validated default
embedding path until the 4B latechunk index is benchmarked.
"""
from __future__ import annotations

import logging
import os

from rag_common.config import LOCAL_MODEL_DIMS, EmbedConfig
from rag_common.local_embed import embed_local_texts_with_tokens, pool_span

from rag_literature_rag.ingest.chunk import TextChunk, assign_late_chunk_windows

log = logging.getLogger("rag_literature_rag.ingest.latechunk")

# How many distinct windows to embed + hold token-level at once before pooling +
# freeing. Bounds memory regardless of how many chunks a single document has — a
# pathological 879-chunk doc has ~100 windows, and embedding/holding them ALL at
# once stalls (giant tensors + .tolist()). Override via env if needed.
WINDOW_EMBED_BATCH = int(os.getenv("RAG_LATECHUNK_WINDOW_BATCH", "8"))


def is_latechunk_profile(profile: str | None) -> bool:
    return bool(profile) and "latechunk" in profile.lower()


def embed_late_chunks(chunks: list[TextChunk], config: EmbedConfig) -> list[list[float]]:
    """Return one late-chunked vector per chunk (input order preserved).

    Embeds each distinct window once at token level, then pools each chunk's
    [window_char_start, window_char_end) span. Falls back to assigning windows
    if they were not set by the caller.
    """
    if not chunks:
        return []
    if any(c.window_id is None or c.window_text is None for c in chunks):
        assign_late_chunk_windows(chunks)

    # Distinct windows in first-seen order + the chunk positions each one feeds.
    window_texts: dict[str, str] = {}
    order: list[str] = []
    positions_by_window: dict[str, list[int]] = {}
    for idx, c in enumerate(chunks):
        wid = c.window_id or c.doc_id
        if wid not in window_texts:
            window_texts[wid] = c.window_text or c.text
            order.append(wid)
        positions_by_window.setdefault(wid, []).append(idx)

    native = LOCAL_MODEL_DIMS.get(config.model, config.dimensions)
    target = config.dimensions if config.dimensions and config.dimensions < native else None

    # Embed + pool in bounded window batches so memory stays flat even for a doc
    # with hundreds of windows (token tensors for the batch are discarded each loop).
    vectors: list[list[float]] = [[] for _ in chunks]
    for start in range(0, len(order), WINDOW_EMBED_BATCH):
        batch_wids = order[start : start + WINDOW_EMBED_BATCH]
        token_data = embed_local_texts_with_tokens(
            [window_texts[w] for w in batch_wids], config=config
        )
        for wid, (tok_vecs, offsets) in zip(batch_wids, token_data):
            if not tok_vecs:
                continue
            for idx in positions_by_window[wid]:
                c = chunks[idx]
                vectors[idx] = pool_span(
                    tok_vecs,
                    offsets,
                    c.window_char_start,
                    c.window_char_end,
                    method="mean",
                    target_dim=target,
                )
    return vectors
