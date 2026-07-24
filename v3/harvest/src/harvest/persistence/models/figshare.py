"""Table-shaped DTO for public.source_figshare + FigshareArticle -> Write mapper.

Mirrors ``persistence.models.openalex``: one row group, keyed by the seed's
bigint ``paper_id``. The typed columns are the queryable subset; the whole
article is kept on ``raw``, with its ``files`` / ``authors`` arrays broken out to
jsonb columns for convenience.

``FigshareArticle`` is imported only for typing — the mapper reads its attributes
and never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.figshare import FigshareArticle


class SourceFigshareRow(TableRow):
    """Insert/upsert shape for public.source_figshare (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``files``, ``authors`` and ``raw`` are
    jsonb; the scalar columns are the promoted, queryable figshare fields.
    ``resource_doi`` is the seed's publication DOI; ``doi`` is figshare's own
    minted deposit DOI (distinct — see the source model docstring).
    """

    __table__ = "source_figshare"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    figshare_id: int | None = None
    doi: str | None = None
    resource_doi: str | None = None
    title: str | None = None
    resource_title: str | None = None
    description: str | None = None
    defined_type_name: str | None = None
    figshare_url: str | None = None
    url_public_html: str | None = None
    published_date: str | None = None
    version: int | None = None
    download_url: str | None = None
    license_name: str | None = None
    tags: list[str] | None = None
    files: list[dict[str, Any]] | None = None
    authors: list[dict[str, Any]] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def figshare_write(
    seed_paper_id: int,
    article: FigshareArticle,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_figshare`` success ``Write`` for one fetched article.

    A single row group keyed by the seed's ``paper_id`` — figshare has no citation
    graph to fan out (contrast ``s2_write``), so there is nothing beyond the one
    outcome row.
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    row = SourceFigshareRow(
        paper_id=seed_paper_id,
        status="succeeded",
        figshare_id=article.id,
        doi=article.doi,
        resource_doi=article.resource_doi,
        title=article.title,
        resource_title=article.resource_title,
        description=article.description,
        defined_type_name=article.defined_type_name,
        figshare_url=article.figshare_url,
        url_public_html=article.url_public_html,
        published_date=article.published_date,
        version=article.version,
        download_url=article.download_url,
        license_name=article.license_name,
        tags=list(article.tags) or None,
        # files / authors already live in `raw`; break them out for convenient querying.
        files=article.raw.get("files"),
        authors=article.raw.get("authors"),
        raw=article.raw,
        fetched_at=now,
    )
    return Write(groups=[[row]])
