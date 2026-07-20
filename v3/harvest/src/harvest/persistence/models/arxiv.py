"""Table-shaped DTO for public.source_arxiv + arxiv.Result -> Write mapper.

Unlike S2 — whose SDK object is normalised into an S2Record DTO first — the arxiv
client's ``Result`` is already a clean, lossless parse, so the mapper consumes it
directly. The nested ``Author``/``Link`` objects are projected to jsonb-friendly
dicts here (the job the old ``ArxivPaper.as_raw()`` did before the DTO was dropped).

``arxiv`` is imported only for typing: the mapper reads ``Result``'s public
attributes and never constructs one, so the persistence layer keeps no runtime
dependency on the source library.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    import arxiv


class SourceArxivRow(TableRow):
    """Insert/upsert shape for public.source_arxiv (per-seed outcome).

    Keyed by the seed's bigint ``paper_id``. ``authors``/``links`` are jsonb
    (nested objects kept whole), matching the lossless table; ``raw`` is the
    faithful projection of ``arxiv.Result`` — the client retains no source XML,
    so this is the most fidelity available. One row per seed records whether the
    arxiv fetch succeeded.
    """

    __table__ = "source_arxiv"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    entry_id: str | None = None
    title: str | None = None
    authors: list[dict[str, Any]] = []
    summary: str | None = None
    primary_category: str | None = None
    categories: list[str] = []
    pdf_url: str | None = None
    links: list[dict[str, Any]] = []
    published: datetime | None = None
    updated: datetime | None = None
    comment: str | None = None
    journal_ref: str | None = None
    doi: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def _authors(result: arxiv.Result) -> list[dict[str, Any]]:
    """``result.authors`` -> ``[{name, affiliation}]``, affiliation kept (not dropped)."""
    return [{"name": a.name, "affiliation": a.affiliation} for a in result.authors]


def _links(result: arxiv.Result) -> list[dict[str, Any]]:
    """``result.links`` -> ``[{href, title, rel, content_type}]``, labels kept whole."""
    return [
        {"href": link.href, "title": link.title, "rel": link.rel, "content_type": link.content_type}
        for link in result.links
    ]


def _raw(result: arxiv.Result, authors: list[dict], links: list[dict]) -> dict[str, Any]:
    """Faithful, json-serialisable projection of every public ``Result`` attribute.

    Datetimes are ISO strings so the dict drops straight into the jsonb column.
    Authors/links are the already-projected dicts (computed once by the caller).
    """
    return {
        "entry_id": result.entry_id,
        "title": result.title,
        "authors": authors,
        "summary": result.summary,
        "primary_category": result.primary_category,
        "categories": list(result.categories),
        "pdf_url": result.pdf_url,
        "links": links,
        "published": result.published.isoformat() if result.published else None,
        "updated": result.updated.isoformat() if result.updated else None,
        "comment": result.comment,
        "journal_ref": result.journal_ref,
        "doi": result.doi,
    }


def arxiv_write(
    seed_paper_id: int,
    result: arxiv.Result,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_arxiv`` success ``Write`` for one fetched ``arxiv.Result``.

    A single row group keyed by the seed's ``paper_id`` — arxiv has no citation
    graph, so there is nothing to fan out (contrast ``s2_write``).
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    authors = _authors(result)
    links = _links(result)
    row = SourceArxivRow(
        paper_id=seed_paper_id,
        status="succeeded",
        entry_id=result.entry_id,
        title=result.title,
        authors=authors,
        summary=result.summary,
        primary_category=result.primary_category,
        categories=list(result.categories),
        pdf_url=result.pdf_url,
        links=links,
        published=result.published,
        updated=result.updated,
        comment=result.comment,
        journal_ref=result.journal_ref,
        doi=result.doi,
        raw=_raw(result, authors, links),
        fetched_at=now,
    )
    return Write(groups=[[row]])
