"""Per-document vectors for litmap, mean-pooled from a tool's champion chunk index.

Distinct from each tool's *citation-relatedness* doc vectors (e.g. GLR's
`graph_layout_rag.doc_vectors`, which embeds title+abstract with a SciNCL/SPECTER2 model
into a small per-model LanceDB table for `cite related`). This module reads the vectors a
tool has *already computed* for retrieval — the `chunks` table's `vector` column in the
champion LanceDB index — and pools them to one vector per doc. No new embedding compute:
it's an I/O + pooling pass, so it should be fast (seconds, not minutes) even at tens of
thousands of chunks, unlike the SPECTER2 path which calls a model.

Doc vector = chunk-count-weighted mean of that doc's chunk vectors (i.e. a plain running
average over chunks — a doc with more chunks naturally weights each chunk equally, which
*is* the chunk-count weighting: summing all chunk vectors and dividing by chunk count),
then L2-normalized so cosine similarity is a dot product.

Persistence is keyed by `(embed_profile, index_fingerprint)` so a stale cache is never
silently reused after a re-index; write to `.npz` (vectors) + a `.json` sidecar (meta).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_NPZ_NAME = "doc_vectors.npz"
_META_NAME = "doc_vectors.meta.json"


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def mean_pool_chunks(
    chunk_vectors: Iterable[tuple[str, list[float]]],
) -> dict[str, list[float]]:
    """Pool `(doc_id, chunk_vector)` pairs into one L2-normalized vector per doc_id.

    A doc with 0 chunks never appears (nothing to average). A doc with 1 chunk yields
    that chunk's vector, normalized. A doc with many chunks yields their mean, normalized.
    """
    sums: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for doc_id, vec in chunk_vectors:
        acc = sums.get(doc_id)
        if acc is None:
            sums[doc_id] = list(vec)
            counts[doc_id] = 1
        else:
            for i, v in enumerate(vec):
                acc[i] += v
            counts[doc_id] += 1
    return {
        doc_id: l2_normalize([x / counts[doc_id] for x in total]) for doc_id, total in sums.items()
    }


def iter_chunk_vectors_from_lancedb(
    lance_dir: Path,
    *,
    table: str = "chunks",
    doc_id_col: str = "doc_id",
    vector_col: str = "vector",
) -> Iterator[tuple[str, list[float]]]:
    """Stream `(doc_id, vector)` pairs from a tool's LanceDB chunk table.

    Imports `lancedb` lazily so this module (and the pure `mean_pool_chunks` above) stays
    importable/testable without the dependency installed.
    """
    import lancedb

    db = lancedb.connect(str(lance_dir))
    table_names = db.list_tables()
    if not isinstance(table_names, list):
        table_names = list(getattr(table_names, "tables", table_names))
    if table not in table_names:
        return
    tbl = db.open_table(table)
    arrow = tbl.to_arrow().select([doc_id_col, vector_col])
    for batch in arrow.to_batches():
        doc_ids = batch.column(doc_id_col).to_pylist()
        vectors = batch.column(vector_col).to_pylist()
        yield from zip(doc_ids, vectors)


def build_doc_vectors_from_lancedb(
    lance_dir: Path,
    *,
    table: str = "chunks",
    progress_cb: Callable[[int], None] | None = None,
) -> dict[str, list[float]]:
    """Build pooled doc vectors straight from a LanceDB chunk index.

    `progress_cb(n_chunks_seen)` is called every 5,000 chunks so a CLI can log progress on
    large corpora; pass `None` (default) to skip.
    """

    def _counted() -> Iterator[tuple[str, list[float]]]:
        n = 0
        for doc_id, vec in iter_chunk_vectors_from_lancedb(lance_dir, table=table):
            n += 1
            if progress_cb is not None and n % 5000 == 0:
                progress_cb(n)
            yield doc_id, vec
        if progress_cb is not None:
            progress_cb(n)

    return mean_pool_chunks(_counted())


def save_doc_vectors(
    out_dir: Path,
    vectors: dict[str, list[float]],
    *,
    embed_profile: str,
    index_fingerprint: str,
) -> None:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = sorted(vectors)
    dim = len(next(iter(vectors.values()))) if vectors else 0
    arr = np.array([vectors[d] for d in doc_ids], dtype=np.float32).reshape(len(doc_ids), dim)
    np.savez(out_dir / _NPZ_NAME, vectors=arr, doc_ids=np.array(doc_ids, dtype=object))
    meta = {
        "schema_version": SCHEMA_VERSION,
        "embed_profile": embed_profile,
        "index_fingerprint": index_fingerprint,
        "doc_count": len(doc_ids),
        "dim": dim,
    }
    (out_dir / _META_NAME).write_text(json.dumps(meta, indent=2))


def load_doc_vectors(
    out_dir: Path,
    *,
    embed_profile: str,
    index_fingerprint: str,
) -> dict[str, list[float]] | None:
    """Load a previously persisted doc-vector cache. `None` on a miss (absent, or the
    profile/fingerprint no longer matches — the caller should rebuild)."""
    meta_path = out_dir / _META_NAME
    npz_path = out_dir / _NPZ_NAME
    if not meta_path.exists() or not npz_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if (
        meta.get("embed_profile") != embed_profile
        or meta.get("index_fingerprint") != index_fingerprint
    ):
        return None
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    doc_ids = list(data["doc_ids"])
    vecs = data["vectors"]
    return {str(doc_id): vecs[i].tolist() for i, doc_id in enumerate(doc_ids)}


def load_doc_vectors_meta(out_dir: Path) -> dict[str, Any] | None:
    meta_path = out_dir / _META_NAME
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())
