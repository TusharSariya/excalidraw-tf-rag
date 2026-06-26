"""Index integrity guards (HG2 chunk parity, HG3 synced-BM25 integrity).

These are fail-closed checks for the cross-machine embedding ladder. They read the
authoritative LanceDB table for chunk counts — NEVER ``ingest_status.json``, which was
observed to report 7124/399 for an index whose real table held 44,672 rows (a ~6x
reporting artifact from a partial run's progress snapshot).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from graph_layout_rag.ingest import bm25
from graph_layout_rag.ingest.index import chunk_count
from graph_layout_rag.paths import ProfileIndexPaths, profile_index_paths

# Files that are NOT byte-stable across a faithful rsync copy between hosts: tantivy's
# managed-file list + meta can carry host/opstamp-specific content, and lock files are
# transient. The segment data files (.idx/.pos/.fast/.fieldnorm/.term/.store/...) ARE
# byte-identical on a faithful copy, so they are what HG3 hashes.
_BM25_HASH_EXCLUDE = {"meta.json", ".managed.json"}


def _bm25_tree_hash(bm25_dir: Path) -> tuple[str, int]:
    """sha256 over sorted (name, size, content-hash) of stable BM25 segment files."""
    if not bm25_dir.is_dir():
        return ("", 0)
    digest = hashlib.sha256()
    n = 0
    for path in sorted(bm25_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        if path.name in _BM25_HASH_EXCLUDE or path.name.endswith(".lock"):
            continue
        data = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(str(len(data)).encode())
        digest.update(hashlib.sha256(data).digest())
        n += 1
    return (digest.hexdigest(), n)


def index_fingerprint(profile: str | ProfileIndexPaths) -> dict[str, Any]:
    """Portable integrity fingerprint of an index, comparable across hosts."""
    paths = profile if isinstance(profile, ProfileIndexPaths) else profile_index_paths(profile)
    bm25_hash, n_files = _bm25_tree_hash(paths.bm25_dir)
    return {
        "profile": paths.profile,
        "chunk_count": chunk_count(paths),
        "bm25_doc_count": bm25.chunk_count(paths.bm25_dir),
        "bm25_tree_hash": bm25_hash,
        "bm25_n_files": n_files,
    }


def assert_chunk_parity(source: str, target: str) -> int:
    """HG2: target index must hold exactly as many chunks as the source. Returns the
    shared count; raises ValueError on mismatch."""
    src_n = chunk_count(source)
    tgt_n = chunk_count(target)
    if src_n == 0:
        raise ValueError(f"HG2 parity: source {source!r} has 0 chunks (missing index?)")
    if src_n != tgt_n:
        raise ValueError(
            f"HG2 parity FAIL: source {source!r}={src_n} chunks but target "
            f"{target!r}={tgt_n}. Target is short/divergent — do not benchmark it."
        )
    return src_n


def compare_fingerprints(local: dict[str, Any], remote: dict[str, Any]) -> list[str]:
    """HG3: return a list of human-readable mismatches (empty == identical).

    Compares chunk count, BM25 doc count, and the BM25 segment tree hash so a
    truncated/partial sync to the remote host is caught before it scores lexically.
    """
    problems: list[str] = []
    for key in ("chunk_count", "bm25_doc_count", "bm25_tree_hash", "bm25_n_files"):
        lv, rv = local.get(key), remote.get(key)
        if lv != rv:
            problems.append(f"{key}: local={lv!r} != remote={rv!r}")
    return problems
