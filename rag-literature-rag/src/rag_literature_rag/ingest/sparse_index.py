"""Impact-weighted learned-sparse inverted index for rag-literature-rag.

The lexical counterpart to :mod:`rag_literature_rag.ingest.bm25`: where BM25 scores
with a term-frequency formula, this index stores per-chunk ``{token_id: weight}``
bags produced by a learned sparse encoder (see
:mod:`rag_common.sparse_encode`) and scores a query bag by a dot product over
shared tokens. It lives at ``data/indexes/<profile>/sparse/`` alongside the
LanceDB and BM25 indexes and returns the SAME candidate-row shape as
``bm25.search_bm25`` so it drops straight into the RRF fusion in
:mod:`rag_literature_rag.query.hybrid`.

Storage — a single **SQLite** file (``sparse/index.sqlite``):

* stdlib only (no numpy/torch to load a query), so cold-load is just opening the
  file — well under the <2s target at the <100k-chunk corpus scale;
* the ``postings(term_id, doc_idx, weight)`` table with an index on ``term_id``
  turns query scoring into one indexed ``WHERE term_id IN (...)`` scan rather than
  reading the whole index into memory;
* natural incremental upsert (delete-by-chunk + insert), matching BM25's
  per-batch re-ingest semantics, so a resumed / ``--force`` run stays cheap.

The ``meta`` table records the encoder ``model_id`` + ``version``, the chunk
count, and an order-independent content ``checksum`` (running XOR of per-chunk
digests). :func:`assert_encoder_matches` uses ``model_id``/``version`` as an
HG1-style guard: a query encoder that does not match the index the postings were
built with is refused rather than silently scoring against a mismatched vocab.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_literature_rag.ingest.chunk import TextChunk

FORMAT_VERSION = 1
_DB_NAME = "index.sqlite"
# SQLite bind-parameter ceiling on older builds is 999; batch IN clauses below it.
_SQL_VAR_LIMIT = 900


@dataclass(frozen=True)
class SparseIndexMeta:
    model_id: str
    version: str
    chunk_count: int
    checksum: str
    format_version: int = FORMAT_VERSION


def sparse_index_db(index_dir: Path) -> Path:
    return index_dir / _DB_NAME


def _chunk_to_row(chunk: TextChunk, body: str) -> dict[str, Any]:
    """Candidate row mirroring ``bm25.search_bm25`` output (minus ``score``).

    ``body`` is the enriched index text (same surface form BM25 stores under
    ``text``), so a learned-sparse hit renders identically to a lexical hit.
    """
    return {
        "id": f"{chunk.doc_id}:{chunk.chunk_index}",
        "doc_id": chunk.doc_id,
        "title": chunk.title or "",
        "text": body,
        "tags": ",".join(chunk.tags),
        "pipeline_categories": ",".join(chunk.pipeline_categories),
        "section_path": chunk.section_path or "",
        "alias_doc_ids": ",".join(chunk.alias_doc_ids),
        "alias_source_urls": ",".join(chunk.alias_source_urls),
        "alias_dois": ",".join(chunk.alias_dois),
        "canonical_sha256": chunk.canonical_sha256 or "",
        "source_url": chunk.source_url or "",
        "year": chunk.year,
        "page": chunk.page,
        "page_end": chunk.page_end,
    }


def _tok_digest(chunk_id: str, tokens: dict[int, float]) -> int:
    """Order-independent 64-bit per-chunk digest for the running checksum."""
    payload = chunk_id + "\x00" + json.dumps(sorted(tokens.items()), separators=(",", ":"))
    # signed=True keeps it inside SQLite's signed 64-bit INTEGER range; XOR combine
    # and the final &-mask below are unaffected by the sign.
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big", signed=True)


def _connect(index_dir: Path, *, create: bool) -> sqlite3.Connection | None:
    db = sparse_index_db(index_dir)
    if not db.exists() and not create:
        return None
    if create:
        index_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS chunks (
            doc_idx  INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE,
            doc_id   TEXT,
            row_json TEXT,
            tok_hash INTEGER
        );
        CREATE TABLE IF NOT EXISTS postings (
            term_id INTEGER,
            doc_idx INTEGER,
            weight  REAL
        );
        CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term_id);
        CREATE INDEX IF NOT EXISTS idx_postings_doc  ON postings(doc_idx);
        """
    )


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def upsert_chunks(
    chunks: list[TextChunk],
    index_texts: list[str],
    token_dicts: list[dict[int, float]],
    *,
    index_dir: Path,
    model_id: str,
    version: str,
    rebuild: bool = False,
) -> int:
    """Insert/replace ``chunks`` in the learned-sparse index.

    ``token_dicts[i]`` is the encoder's ``{token_id: weight}`` bag for
    ``index_texts[i]`` (the same enriched body BM25 indexes). Refuses to mix
    encoders: an existing index built with a different ``model_id``/``version``
    must be ``rebuild``-ed.
    """
    if not (len(chunks) == len(index_texts) == len(token_dicts)):
        raise ValueError(
            f"length mismatch chunks={len(chunks)} texts={len(index_texts)} "
            f"tokens={len(token_dicts)}"
        )
    if rebuild:
        sparse_index_db(index_dir).unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(sparse_index_db(index_dir)) + suffix).unlink(missing_ok=True)
    if not chunks:
        return 0

    conn = _connect(index_dir, create=True)
    assert conn is not None
    try:
        _init_schema(conn)
        stored_model = _meta_get(conn, "model_id")
        stored_version = _meta_get(conn, "version")
        if stored_model is not None and (stored_model != model_id or stored_version != version):
            raise RuntimeError(
                f"sparse index at {index_dir} was built with "
                f"{stored_model!r}/{stored_version!r} but this run uses "
                f"{model_id!r}/{version!r}; re-run with --rebuild."
            )
        checksum = int(_meta_get(conn, "checksum") or "0", 16)

        for chunk, body, tokens in zip(chunks, index_texts, token_dicts):
            row = _chunk_to_row(chunk, body)
            chunk_id = row["id"]
            digest = _tok_digest(chunk_id, tokens)
            existing = conn.execute(
                "SELECT doc_idx, tok_hash FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            if existing is not None:
                doc_idx, old_hash = existing
                checksum ^= int(old_hash)
                conn.execute("DELETE FROM postings WHERE doc_idx = ?", (doc_idx,))
                conn.execute(
                    "UPDATE chunks SET doc_id = ?, row_json = ?, tok_hash = ? WHERE doc_idx = ?",
                    (row["doc_id"], json.dumps(row), digest, doc_idx),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO chunks(chunk_id, doc_id, row_json, tok_hash) VALUES(?, ?, ?, ?)",
                    (chunk_id, row["doc_id"], json.dumps(row), digest),
                )
                doc_idx = cur.lastrowid
            checksum ^= digest
            if tokens:
                conn.executemany(
                    "INSERT INTO postings(term_id, doc_idx, weight) VALUES(?, ?, ?)",
                    [(int(tid), doc_idx, float(w)) for tid, w in tokens.items()],
                )

        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        _meta_set(conn, "model_id", model_id)
        _meta_set(conn, "version", version)
        _meta_set(conn, "format_version", str(FORMAT_VERSION))
        _meta_set(conn, "chunk_count", str(count))
        _meta_set(conn, "checksum", f"{checksum & ((1 << 64) - 1):016x}")
        conn.commit()
        return len(chunks)
    finally:
        conn.close()


def load_meta(index_dir: Path) -> SparseIndexMeta | None:
    conn = _connect(index_dir, create=False)
    if conn is None:
        return None
    try:
        if _meta_get(conn, "model_id") is None:
            return None
        return SparseIndexMeta(
            model_id=_meta_get(conn, "model_id") or "",
            version=_meta_get(conn, "version") or "",
            chunk_count=int(_meta_get(conn, "chunk_count") or "0"),
            checksum=_meta_get(conn, "checksum") or "",
            format_version=int(_meta_get(conn, "format_version") or "0"),
        )
    finally:
        conn.close()


def chunk_count(index_dir: Path) -> int:
    meta = load_meta(index_dir)
    return meta.chunk_count if meta else 0


def assert_encoder_matches(index_dir: Path, encoder: Any) -> None:
    """HG1-style guard: refuse a query encoder that does not match the index."""
    meta = load_meta(index_dir)
    if meta is None:
        raise RuntimeError(
            f"No learned-sparse index at {index_dir}. Build it with a sparse-enabled ingest."
        )
    if meta.model_id != encoder.model_id or meta.version != encoder.version:
        raise RuntimeError(
            f"Learned-sparse index at {index_dir} was built with "
            f"{meta.model_id!r}/{meta.version!r} but the query encoder is "
            f"{encoder.model_id!r}/{encoder.version!r}. Refusing to score against a "
            f"mismatched sparse model — rebuild the index or select the matching encoder."
        )


def score(
    query_tokens: dict[int, float],
    *,
    index_dir: Path,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Dot-product score ``query_tokens`` against the index; top-``limit`` rows.

    Returns candidate rows in the same shape as ``bm25.search_bm25`` with a
    ``score`` field (the accumulated impact dot product).
    """
    conn = _connect(index_dir, create=False)
    if conn is None or not query_tokens:
        if conn is not None:
            conn.close()
        return []
    try:
        terms = [int(t) for t, w in query_tokens.items() if float(w) > 0.0]
        scores: dict[int, float] = {}
        for start in range(0, len(terms), _SQL_VAR_LIMIT):
            batch = terms[start : start + _SQL_VAR_LIMIT]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT term_id, doc_idx, weight FROM postings WHERE term_id IN ({placeholders})",
                batch,
            )
            for term_id, doc_idx, weight in rows:
                scores[doc_idx] = scores.get(doc_idx, 0.0) + query_tokens[term_id] * weight
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        by_idx: dict[int, dict[str, Any]] = {}
        idxs = [doc_idx for doc_idx, _ in top]
        for start in range(0, len(idxs), _SQL_VAR_LIMIT):
            batch = idxs[start : start + _SQL_VAR_LIMIT]
            placeholders = ",".join("?" * len(batch))
            for doc_idx, row_json in conn.execute(
                f"SELECT doc_idx, row_json FROM chunks WHERE doc_idx IN ({placeholders})",
                batch,
            ):
                by_idx[doc_idx] = json.loads(row_json)
        results: list[dict[str, Any]] = []
        for doc_idx, sc in top:
            row = by_idx.get(doc_idx)
            if row is None:
                continue
            row = dict(row)
            row["score"] = round(float(sc), 6)
            results.append(row)
        return results
    finally:
        conn.close()
