"""Citation-graph coverage audit, shared by graph-layout-rag and rag-literature-rag.

Both tools' `citations.sqlite` share the same `papers(oa_id, doi, doc_id, title, year,
cited_by_count, in_corpus, ...)` / `cites(src_oa, dst_oa, is_influential)` core schema
(see each tool's `citation_store.py`); GLR additionally has `papers_meta`/`doc_specter2`
tables this module doesn't touch. Pure stdlib `sqlite3` — no tool-specific imports — so
it lives here and both tools' `cli.py` wire it into their `cite coverage` subcommand.

This is the E0 audit artifact: how much of the corpus is actually keyed into the citation
graph (vs. papers rows that exist only as external/uncited neighbors), so a "citation
signal" isn't silently near-empty. Re-run after every `cite enrich` / keying pass.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CoverageStats:
    """Snapshot of how well `citations.sqlite` is keyed to the corpus manifest."""

    total_corpus_docs: int
    keyed_docs: int
    keyed_pct: float
    total_edges: int
    both_endpoint_keyed_edges: int
    both_endpoint_keyed_pct: float
    keyed_by_decade: dict[str, int] = field(default_factory=dict)
    duplicate_doc_ids: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_corpus_docs": self.total_corpus_docs,
            "keyed_docs": self.keyed_docs,
            "keyed_pct": round(self.keyed_pct, 4),
            "total_edges": self.total_edges,
            "both_endpoint_keyed_edges": self.both_endpoint_keyed_edges,
            "both_endpoint_keyed_pct": round(self.both_endpoint_keyed_pct, 4),
            "keyed_by_decade": self.keyed_by_decade,
            "duplicate_doc_ids": self.duplicate_doc_ids,
        }


def _pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100.0) if denominator else 0.0


def compute_coverage(db: sqlite3.Connection, *, total_corpus_docs: int) -> CoverageStats:
    """Compute coverage stats from an open `citations.sqlite` connection.

    ``total_corpus_docs`` is the manifest item count — the denominator for "% of corpus
    keyed" — supplied by the caller (each tool's `load_manifest()`) so this module needs
    no tool-specific import.
    """
    keyed_docs = db.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM papers WHERE doc_id IS NOT NULL"
    ).fetchone()[0]

    total_edges = db.execute("SELECT COUNT(*) FROM cites").fetchone()[0]
    both_keyed = db.execute(
        """
        SELECT COUNT(*) FROM cites c
        JOIN papers p1 ON c.src_oa = p1.oa_id
        JOIN papers p2 ON c.dst_oa = p2.oa_id
        WHERE p1.doc_id IS NOT NULL AND p2.doc_id IS NOT NULL
        """
    ).fetchone()[0]

    decade_rows = db.execute(
        """
        SELECT (year / 10) * 10 AS decade, COUNT(DISTINCT doc_id) AS n
        FROM papers
        WHERE doc_id IS NOT NULL AND year IS NOT NULL
        GROUP BY decade
        ORDER BY decade
        """
    ).fetchall()
    keyed_by_decade = {f"{int(row[0])}s": int(row[1]) for row in decade_rows}
    undated = db.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM papers WHERE doc_id IS NOT NULL AND year IS NULL"
    ).fetchone()[0]
    if undated:
        keyed_by_decade["undated"] = undated

    dup_rows = db.execute(
        """
        SELECT doc_id, COUNT(*) AS n
        FROM papers
        WHERE doc_id IS NOT NULL
        GROUP BY doc_id
        HAVING COUNT(*) > 1
        ORDER BY n DESC, doc_id
        """
    ).fetchall()
    duplicate_doc_ids = {row[0]: int(row[1]) for row in dup_rows}

    return CoverageStats(
        total_corpus_docs=total_corpus_docs,
        keyed_docs=keyed_docs,
        keyed_pct=_pct(keyed_docs, total_corpus_docs),
        total_edges=total_edges,
        both_endpoint_keyed_edges=both_keyed,
        both_endpoint_keyed_pct=_pct(both_keyed, total_edges),
        keyed_by_decade=keyed_by_decade,
        duplicate_doc_ids=duplicate_doc_ids,
    )
