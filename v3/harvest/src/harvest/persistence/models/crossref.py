"""Table-shaped DTOs for public.source_crossref + public.crossref_reference,
plus the CrossrefWork -> Write mapper.

Mirrors ``persistence.models.s2``: a multi-table fan-out. The subject work's
per-seed OUTCOME lands on ``source_crossref`` (keyed by the seed's bigint
``paper_id``); its ``reference[]`` list fans out to ``crossref_reference`` edge
rows. Both tables FK only into ``seeds`` — there is no Crossref node table — so
the edge group is emitted BEFORE the outcome row (FK order: parents first), same
ordering discipline as ``s2_write``.

``CrossrefWork`` is imported only for typing — the mapper reads its attributes and
never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.crossref import CrossrefWork


class CrossrefReferenceRow(TableRow):
    """Insert/upsert shape for public.crossref_reference (one outbound reference edge).

    Keyed by ``(paper_id, ref_key)`` — ``ref_key`` is Crossref's per-work
    ``reference[].key``, which is always present and unique within a work.
    ``target_doi`` is the reference's ``DOI`` when Crossref carries one (~30-70%
    of entries), else None (unstructured / DOI-less reference). The whole
    reference entry is kept on ``raw``.
    """

    __table__ = "crossref_reference"
    __conflict__ = ("paper_id", "ref_key")

    paper_id: int
    ref_key: str
    target_doi: str | None = None
    doi_asserted_by: str | None = None
    article_title: str | None = None
    journal_title: str | None = None
    author: str | None = None
    year: str | None = None
    volume: str | None = None
    first_page: str | None = None
    unstructured: str | None = None
    raw: dict[str, Any] | None = None


class SourceCrossrefRow(TableRow):
    """Insert/upsert shape for public.source_crossref (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``authors`` and ``raw`` are jsonb; the
    scalar columns are the promoted, queryable Crossref fields. ``doi`` is stored
    as Crossref returns it (lowercased).
    """

    __table__ = "source_crossref"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    doi: str | None = None
    title: str | None = None
    container_title: str | None = None
    type: str | None = None
    publisher: str | None = None
    url: str | None = None
    issued_year: int | None = None
    is_referenced_by_count: int | None = None
    references_count: int | None = None
    page: str | None = None
    issn: list[str] | None = None
    authors: list[dict[str, Any]] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def _reference_rows(seed_paper_id: int, work: CrossrefWork) -> list[CrossrefReferenceRow]:
    """One ``crossref_reference`` row per ``reference[]`` entry, deduped by ref_key.

    Entries without a ``key`` are skipped — ``ref_key`` is the edge's natural key
    and Crossref always supplies it in practice.
    """
    rows: dict[str, CrossrefReferenceRow] = {}
    for ref in work.references:
        if not ref.key:
            continue
        rows.setdefault(
            ref.key,
            CrossrefReferenceRow(
                paper_id=seed_paper_id,
                ref_key=ref.key,
                target_doi=ref.doi,
                doi_asserted_by=ref.doi_asserted_by,
                article_title=ref.article_title,
                journal_title=ref.journal_title,
                author=ref.author,
                year=ref.year,
                volume=ref.volume,
                first_page=ref.first_page,
                unstructured=ref.unstructured,
                raw=ref.raw or None,
            ),
        )
    return list(rows.values())


def crossref_write(
    seed_paper_id: int,
    work: CrossrefWork,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the full Crossref persistence payload for one fetched ``CrossrefWork``.

    The ``crossref_reference`` edge rows (FK-ordered first) followed by the
    per-seed ``source_crossref`` outcome row, as one ``Write``. Both groups FK
    only into ``seeds`` (already present); the edge group leads so it mirrors the
    parents-before-children discipline of ``s2_write``. Empty groups are dropped.
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    edges = _reference_rows(seed_paper_id, work)
    outcome = SourceCrossrefRow(
        paper_id=seed_paper_id,
        status="succeeded",
        doi=work.doi or None,
        title=work.title,
        container_title=work.container_title,
        type=work.type,
        publisher=work.publisher,
        url=work.url,
        issued_year=work.issued_year,
        is_referenced_by_count=work.is_referenced_by_count,
        references_count=work.references_count,
        page=work.page,
        issn=work.issn or None,
        # author already lives in `raw`; break it out for convenient querying.
        authors=work.raw.get("author"),
        raw=work.raw,
        fetched_at=now,
    )
    groups: list[Sequence[TableRow]] = [edges, [outcome]]
    return Write(groups=[g for g in groups if g])
