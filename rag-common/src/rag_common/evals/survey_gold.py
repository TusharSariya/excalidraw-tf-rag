"""Gold-set builder for the reference-concentration eval: find in-corpus surveys and their
in-corpus reference lists from a tool's `citations.sqlite` + manifest.

Pure `sqlite3` stdlib + plain dicts — no tool import, mirrors `rag_common.citation_coverage`
(same `papers`/`cites` schema assumption: `papers(oa_id, doi, doc_id, title, year, ...,
in_corpus)`, `cites(src_oa, dst_oa, ...)`).

Survey detection (D13/plan): a doc qualifies as a "survey" gold case if its title/tags
contain a survey-ish keyword, OR it has >=`min_refs` in-corpus outgoing references (a
paper that cites 100+ other corpus docs is landscape-spanning regardless of what its title
says). Per-section gold (survey section -> its own ref list) is legitimately better —
surveys span clusters section by section — but extracting section-scoped reference
contexts needs the ingest chunk store to carry `section_path` on citation contexts, which
it doesn't yet; this module implements the doc-level granularity (single `"_all"` section)
now, behind the same `{survey_doc_id: {section_path: [ref_doc_id,...]}}` shape
`concentration.concentration_report` expects, so wiring in real per-section extraction
later is a pure data change, not an API change.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MIN_REFS = 100
DEFAULT_KEYWORDS = ("survey", "state of the art", "taxonomy")
# Fallback keyword set (plan: "GLR fallback to handbook/monograph chapters only if <5
# qualify") — broader net, opt-in via `find_surveys(..., use_fallback_keywords=True)`.
FALLBACK_KEYWORDS = ("handbook", "monograph", "chapter")


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SurveyGold:
    survey_doc_id: str
    title: str
    sections: dict[str, list[str]]  # {"_all": [ref_doc_id, ...]} at doc-level granularity
    matched_by: str  # "keyword" | "ref_count" | "both"

    @property
    def ref_count(self) -> int:
        return len(self.sections.get("_all", []))

    def to_dict(self) -> dict[str, Any]:
        return {
            "survey_doc_id": self.survey_doc_id,
            "title": self.title,
            "ref_count": self.ref_count,
            "matched_by": self.matched_by,
            "sections": self.sections,
        }


def is_survey_by_text(
    title: str, tags: tuple[str, ...] | list[str], keywords: tuple[str, ...]
) -> bool:
    text = f"{title} {' '.join(tags)}".lower()
    return any(kw in text for kw in keywords)


def in_corpus_refs_by_doc(db: sqlite3.Connection) -> dict[str, list[str]]:
    """`{src_doc_id: [dst_doc_id, ...]}` for every citation edge where both endpoints are
    keyed into the corpus manifest (`papers.doc_id IS NOT NULL`)."""
    rows = db.execute(
        """
        SELECT DISTINCT p1.doc_id AS src_doc, p2.doc_id AS dst_doc
        FROM cites c
        JOIN papers p1 ON c.src_oa = p1.oa_id
        JOIN papers p2 ON c.dst_oa = p2.oa_id
        WHERE p1.doc_id IS NOT NULL AND p2.doc_id IS NOT NULL AND p1.doc_id != p2.doc_id
        """
    ).fetchall()
    out: dict[str, list[str]] = {}
    for src_doc, dst_doc in rows:
        out.setdefault(src_doc, []).append(dst_doc)
    return out


def find_surveys(
    db: sqlite3.Connection,
    doc_meta: dict[str, DocMeta],
    *,
    min_refs: int = DEFAULT_MIN_REFS,
    use_fallback_keywords: bool = False,
) -> list[SurveyGold]:
    """Every in-corpus doc that qualifies as a survey gold case, keyword- or
    reference-count-matched, in `(-ref_count, survey_doc_id)` order.

    ``doc_meta`` maps doc_id -> `DocMeta` (title/tags) sourced from the tool's manifest —
    this module never imports a tool package, so the caller (`cli.py`) builds this dict.
    Docs with no `doc_meta` entry can still qualify via ref-count alone (title/tags default
    to empty text, which never keyword-matches).
    """
    keywords = DEFAULT_KEYWORDS + (FALLBACK_KEYWORDS if use_fallback_keywords else ())
    refs_by_doc = in_corpus_refs_by_doc(db)

    surveys: list[SurveyGold] = []
    for doc_id, refs in refs_by_doc.items():
        meta = doc_meta.get(doc_id, DocMeta(doc_id=doc_id))
        by_keyword = is_survey_by_text(meta.title, meta.tags, keywords)
        by_count = len(refs) >= min_refs
        if not (by_keyword or by_count):
            continue
        matched_by = (
            "both" if (by_keyword and by_count) else ("keyword" if by_keyword else "ref_count")
        )
        surveys.append(
            SurveyGold(
                survey_doc_id=doc_id,
                title=meta.title,
                sections={"_all": sorted(set(refs))},
                matched_by=matched_by,
            )
        )
    surveys.sort(key=lambda s: (-s.ref_count, s.survey_doc_id))
    return surveys


def to_gold_dict(surveys: list[SurveyGold]) -> dict[str, dict[str, list[str]]]:
    """Adapt a `find_surveys` result to the shape `concentration.concentration_report`
    expects: `{survey_doc_id: {section_path: [ref_doc_id, ...]}}`."""
    return {s.survey_doc_id: s.sections for s in surveys}
