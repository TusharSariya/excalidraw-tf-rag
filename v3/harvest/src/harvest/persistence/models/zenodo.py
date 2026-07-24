"""Table-shaped DTO for public.source_zenodo + ZenodoRecord -> Write mapper.

Mirrors ``persistence.models.unpaywall`` / ``persistence.models.openalex``: one
row group, keyed by the seed's bigint ``paper_id``. The typed columns are the
queryable subset; the whole Zenodo hit is kept on ``raw``, with its ``creators``
and ``files`` arrays broken out to jsonb columns for convenience.

``ZenodoRecord`` is imported only for typing — the mapper reads its attributes and
never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.zenodo import ZenodoRecord


class SourceZenodoRow(TableRow):
    """Insert/upsert shape for public.source_zenodo (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``creators``, ``files`` and ``raw`` are
    jsonb; the scalar columns are the promoted, queryable Zenodo fields.
    """

    __table__ = "source_zenodo"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    record_id: int | None = None
    doi: str | None = None
    conceptdoi: str | None = None
    doi_url: str | None = None
    title: str | None = None
    publication_date: str | None = None
    access_right: str | None = None
    version: str | None = None
    resource_type_type: str | None = None
    license_id: str | None = None
    file_download_url: str | None = None
    landing_page: str | None = None
    creators: list[dict[str, Any]] | None = None
    files: list[dict[str, Any]] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def zenodo_write(
    seed_paper_id: int,
    rec: ZenodoRecord,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_zenodo`` success ``Write`` for one fetched ``ZenodoRecord``.

    A single row group keyed by the seed's ``paper_id`` — Zenodo has no citation
    graph, so there is nothing to fan out (contrast ``s2_write``).
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    row = SourceZenodoRow(
        paper_id=seed_paper_id,
        status="succeeded",
        record_id=rec.record_id,
        doi=rec.doi,
        conceptdoi=rec.conceptdoi,
        doi_url=rec.doi_url,
        title=rec.title,
        publication_date=rec.publication_date,
        access_right=rec.access_right,
        version=rec.version,
        resource_type_type=rec.resource_type_type,
        license_id=rec.license_id,
        file_download_url=rec.file_download_url,
        landing_page=rec.landing_page,
        # creators / files already live in `raw`; break them out for convenient querying.
        creators=rec.raw.get("metadata", {}).get("creators"),
        files=rec.raw.get("files"),
        raw=rec.raw,
        fetched_at=now,
    )
    return Write(groups=[[row]])
