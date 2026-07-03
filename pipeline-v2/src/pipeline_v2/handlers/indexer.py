"""Indexer stage handler body: embeddings artifact -> v2 LanceDB table + tantivy index.

This module owns pipeline-v2's *own* index directory tree — never v1's
``graph-layout-rag/data/indexes`` — so the two pipelines can run side by
side without either one corrupting the other's index. Library choice
mirrors v1 exactly (``lancedb``/``tantivy``, pinned in ``pyproject.toml`` to
the same versions graph-layout-rag's lockfile resolves: ``lancedb==0.33.0``,
``tantivy==0.26.0``) but the write path here is intentionally *not* a reuse
of ``graph_layout_rag.ingest.index``/``ingest.bm25`` — those modules carry a
lot of v1-specific state (``ingest_state.json`` resume checkpoints,
``EmbedConfig``/embed-cost accounting, profile-keyed index directories) this
stage has no use for. What's reused is the on-disk row shape (comma-joined
list columns, sentinel ints for nullable tantivy fields) so a v1-style
reader would recognize the schema.

Single-writer discipline: a module-level lock serializes LanceDB/tantivy
writes within one process. The messaging-backbone layer (share-group
consumer, one process per stage) is the actual concurrency boundary — this
lock only protects against this process's own handler being re-entered
concurrently (e.g. a future multi-threaded consumer), not against a second
*process* also writing; the artifact-store's per-message content-addressing
plus per-doc delete-then-insert (idempotent replace, see ``_write_lance``/
``_write_tantivy``) is what keeps a rerun of the same message from
duplicating index rows.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import tantivy

from pipeline_v2.artifacts import ArtifactStore, artifact_store_from_env
from pipeline_v2.consumers import HandlerResult
from pipeline_v2.contracts import StageMessage

logger = logging.getLogger(__name__)

CHUNKS_TABLE = "chunks"
_INDEXES_DIR_ENV = "PIPELINE_V2_INDEXES_DIR"
_NO_PAGE = -1
_write_lock = threading.Lock()


def _default_indexes_root() -> Path:
    raw = os.getenv(_INDEXES_DIR_ENV, "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else Path.cwd() / path
    return Path(__file__).resolve().parents[2] / "data" / "indexes"


@dataclass(frozen=True)
class V2IndexPaths:
    root: Path
    lance_dir: Path
    bm25_dir: Path

    @classmethod
    def for_corpus(cls, corpus: str, root: Path | None = None) -> V2IndexPaths:
        base = (root if root is not None else _default_indexes_root()) / corpus
        return cls(root=base, lance_dir=base / "lancedb", bm25_dir=base / "bm25")


def _chunk_row(doc_id: str, corpus: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{doc_id}:{record.get('chunk_index', 0)}",
        "doc_id": doc_id,
        "corpus": corpus,
        "chunk_index": int(record.get("chunk_index") or 0),
        "title": str(record.get("title") or ""),
        "text": str(record.get("text") or ""),
        "page": record.get("page") if record.get("page") is not None else _NO_PAGE,
        "page_end": record.get("page_end") if record.get("page_end") is not None else _NO_PAGE,
        "section_path": str(record.get("section_path") or ""),
        "source_url": str(record.get("source_url") or ""),
        "vector": record["vector"],
    }


def _table_names(db: lancedb.DBConnection) -> list[str]:
    lister = getattr(db, "list_tables", None) or db.table_names
    names = lister()
    return list(getattr(names, "tables", names))


def _write_lance(lance_dir: Path, doc_id: str, corpus: str, rows: list[dict[str, Any]]) -> int:
    lance_dir.parent.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lance_dir))
    if CHUNKS_TABLE in _table_names(db):
        table = db.open_table(CHUNKS_TABLE)
        # Idempotent replace: drop any prior rows for this exact doc before
        # re-adding, so a message redelivery/rerun never duplicates rows.
        table.delete(f"doc_id = '{doc_id}' AND corpus = '{corpus}'")
        table.add(rows)
    else:
        table = db.create_table(CHUNKS_TABLE, data=rows)
    version = getattr(table, "version", None)
    if callable(version):
        version = version()
    return version if isinstance(version, int) else -1


def _build_bm25_schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("id", stored=True)
    builder.add_text_field("doc_id", stored=True)
    builder.add_text_field("corpus", stored=True)
    builder.add_text_field("title", stored=True)
    builder.add_text_field("text", stored=True)
    builder.add_text_field("section_path", stored=True)
    builder.add_text_field("source_url", stored=True)
    builder.add_integer_field("chunk_index", stored=True)
    builder.add_integer_field("page", stored=True)
    return builder.build()


def _open_or_create_bm25(bm25_dir: Path) -> tantivy.Index:
    bm25_dir.mkdir(parents=True, exist_ok=True)
    return tantivy.Index(_build_bm25_schema(), path=str(bm25_dir))


def _bm25_add(writer: tantivy.IndexWriter, row: dict[str, Any]) -> None:
    writer.add_document(
        tantivy.Document(
            id=row["id"],
            doc_id=row["doc_id"],
            corpus=row["corpus"],
            title=row["title"],
            text=row["text"],
            section_path=row["section_path"],
            source_url=row["source_url"],
            chunk_index=row["chunk_index"],
            page=row["page"],
        )
    )


def _is_bm25_corruption(exc: Exception) -> bool:
    """tantivy leaves a non-crash-atomic index: if the indexer pod is killed
    mid-commit/merge, meta.json can reference a segment whose files are gone,
    and every later open/commit fails with FileDoesNotExist on a .term/.idx/...
    segment file. Detect that class so we can self-heal from LanceDB rather
    than dead-letter every subsequent doc (found during the E15 full ingest,
    planted by the E13/E15 indexer kills)."""
    msg = str(exc).lower()
    # A killed indexer leaves the tantivy index inconsistent; the next
    # open/commit surfaces it one of two ways depending on which file the
    # merge/read touches: a missing segment file ("FileDoesNotExist"/"no such
    # file") or a half-written segment ("Data corruption ... composite file").
    if "filedoesnotexist" in msg or "data corrupt" in msg:
        return True
    seg_ext = (".term", ".idx", ".store", ".fieldnorm", ".pos", ".fast")
    return ("no such file" in msg or "does not exist" in msg) and any(e in msg for e in seg_ext)


def _rebuild_bm25_from_lance(bm25_dir: Path, lance_dir: Path) -> None:
    """Crash-recovery: wipe the corrupted tantivy index and rebuild it from the
    LanceDB chunks table (the source of truth — the Lance write in _index_one
    runs BEFORE the bm25 write, so it already contains the current doc). This
    makes the indexer self-heal from a mid-merge kill instead of getting stuck
    dead-lettering forever."""
    import shutil

    logger.warning("bm25 index at %s is corrupted; rebuilding from LanceDB", bm25_dir)
    if bm25_dir.exists():
        shutil.rmtree(bm25_dir)
    bm25_dir.mkdir(parents=True, exist_ok=True)
    index = tantivy.Index(_build_bm25_schema(), path=str(bm25_dir))
    writer = index.writer()
    db = lancedb.connect(str(lance_dir))
    if CHUNKS_TABLE in _table_names(db):
        table = db.open_table(CHUNKS_TABLE)
        for row in table.to_arrow().to_pylist():
            _bm25_add(writer, row)
    writer.commit()
    index.reload()


def _write_bm25(
    bm25_dir: Path,
    doc_id: str,
    corpus: str,
    rows: list[dict[str, Any]],
    lance_dir: Path | None = None,
) -> int:
    try:
        index = _open_or_create_bm25(bm25_dir)
        writer = index.writer()
        writer.delete_documents("doc_id", doc_id)
        for row in rows:
            _bm25_add(writer, row)
        writer.commit()
        index.reload()
        return len(rows)
    except Exception as exc:  # noqa: BLE001 — narrowed to corruption below; re-raised otherwise
        if lance_dir is not None and _is_bm25_corruption(exc):
            # rows are already in LanceDB (written first in _index_one), so a
            # full rebuild from Lance includes this doc — recover and move on.
            _rebuild_bm25_from_lance(bm25_dir, lance_dir)
            return len(rows)
        raise


def lance_doc_chunk_ids(lance_dir: Path, corpus: str | None = None) -> set[str]:
    """Read back every chunk ``id`` currently in the v2 Lance table (test/debug helper)."""
    if not lance_dir.exists():
        return set()
    db = lancedb.connect(str(lance_dir))
    if CHUNKS_TABLE not in _table_names(db):
        return set()
    table = db.open_table(CHUNKS_TABLE)
    rows = table.to_arrow().to_pylist()
    return {row["id"] for row in rows if corpus is None or row.get("corpus") == corpus}


def bm25_doc_ids(bm25_dir: Path) -> set[str]:
    """Read back every distinct ``doc_id`` currently in the v2 tantivy index (test/debug helper)."""
    if not bm25_dir.exists():
        return set()
    index = tantivy.Index.open(str(bm25_dir))
    index.reload()
    searcher = index.searcher()
    query = index.parse_query("*", ["doc_id"]) if searcher.num_docs > 0 else None
    doc_ids: set[str] = set()
    if query is not None:
        hits = searcher.search(query, limit=max(searcher.num_docs, 1)).hits
        for _score, address in hits:
            doc = searcher.doc(address)
            values = doc.get_first("doc_id")
            if values is not None:
                doc_ids.add(values)
    return doc_ids


def make_index_writer(
    store: ArtifactStore | None = None,
    *,
    indexes_root: Path | None = None,
):
    """Build the callable used *after* the indexer's structural gate passes.

    Returns a ``StageMessage -> HandlerResult`` callable, matching
    ``pipeline_v2.consumers.Handler``'s shape, for
    ``make_indexer_handler(repo, index_writer=...)`` to invoke once the
    upstream-embed gate check succeeds.
    """
    artifact_store = store if store is not None else artifact_store_from_env()
    root = indexes_root

    def _writer(message: StageMessage) -> HandlerResult:
        return _index_one(message, artifact_store, root)

    return _writer


def _index_one(
    message: StageMessage, store: ArtifactStore, indexes_root_override: Path | None
) -> HandlerResult:
    artifact_uri = message.payload.get("embeddings_artifact_uri")
    if not artifact_uri:
        return HandlerResult(state="degraded", reason="payload missing 'embeddings_artifact_uri'")

    try:
        raw = store.get(artifact_uri)
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return HandlerResult(state="degraded", reason=f"could not read embeddings artifact: {exc}")

    chunks = data.get("chunks") or []
    if not chunks:
        return HandlerResult(state="degraded", reason="embeddings artifact has no chunks")

    missing_vectors = [c for c in chunks if not c.get("vector")]
    if missing_vectors:
        return HandlerResult(
            state="degraded",
            reason=f"{len(missing_vectors)} of {len(chunks)} chunks are missing vectors",
        )

    paths = V2IndexPaths.for_corpus(message.corpus, root=indexes_root_override)
    rows = [_chunk_row(message.doc_id, message.corpus, c) for c in chunks]

    with _write_lock:
        lance_version = _write_lance(paths.lance_dir, message.doc_id, message.corpus, rows)
        bm25_count = _write_bm25(
            paths.bm25_dir, message.doc_id, message.corpus, rows, lance_dir=paths.lance_dir
        )

    # No dedicated "lance table version" column exists on WorkStateRow (only
    # doc_id/corpus/stage/state/reason/artifact_uri/run_id) and `run_id` is
    # semantically the *pipeline run* id, not an index-table version — reusing
    # it here would silently corrupt that field's real meaning. Surfacing the
    # version in `reason` is a trivial, non-invasive way to keep it visible
    # without a schema change; a real version->row binding needs a dedicated
    # column. TODO(P3): add that column (or fold it into artifact_uri) if
    # rollback-by-lance-version ever becomes a real requirement.
    return HandlerResult(
        state="ok",
        artifact_uri=artifact_uri,
        reason=f"lance_version={lance_version}",
        output_payload={
            "indexed_chunk_count": bm25_count,
            "lance_table_version": lance_version,
        },
    )
