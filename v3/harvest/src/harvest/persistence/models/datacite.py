"""Table-shaped DTO for public.source_datacite + DataCiteRecord -> Write mapper.

Mirrors ``persistence.models.openalex``: one row group, keyed by the seed's
bigint ``paper_id``. The typed columns are the queryable subset (DOI, resource
type, publisher, license-relevant scalars); the whole JSON:API record is kept on
``raw``, with its ``creators`` array broken out to the ``creators`` jsonb column
for convenience.

``DataCiteRecord`` is imported only for typing — the mapper reads its attributes
and never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.datacite import DataCiteRecord


class SourceDataCiteRow(TableRow):
    """Insert/upsert shape for public.source_datacite (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``creators`` and ``raw`` are jsonb; the
    scalar columns are the promoted, queryable DataCite fields.
    """

    __table__ = "source_datacite"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    doi: str | None = None
    prefix: str | None = None
    suffix: str | None = None
    state: str | None = None
    title: str | None = None
    publisher: str | None = None
    publication_year: int | None = None
    resource_type_general: str | None = None
    resource_type: str | None = None
    url: str | None = None
    version: str | None = None
    language: str | None = None
    schema_version: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    is_active: bool | None = None
    registered: datetime | None = None
    created: datetime | None = None
    updated: datetime | None = None
    creators: list[dict[str, Any]] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def datacite_write(
    seed_paper_id: int,
    record: DataCiteRecord,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_datacite`` success ``Write`` for one fetched ``DataCiteRecord``.

    A single row group keyed by the seed's ``paper_id`` — DataCite carries a
    ``relatedIdentifiers`` edge set, but it is kept in ``raw`` rather than fanned
    into per-datacite graph tables, so there is nothing to fan out here (contrast
    ``s2_write``).
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    row = SourceDataCiteRow(
        paper_id=seed_paper_id,
        status="succeeded",
        doi=record.doi or None,
        prefix=record.prefix,
        suffix=record.suffix,
        state=record.state,
        title=record.title,
        publisher=record.publisher,
        publication_year=record.publication_year,
        resource_type_general=record.resource_type_general,
        resource_type=record.resource_type or None,
        url=record.url,
        version=record.version,
        language=record.language,
        schema_version=record.schema_version,
        citation_count=record.citation_count,
        reference_count=record.reference_count,
        is_active=record.is_active,
        registered=record.registered,
        created=record.created,
        updated=record.updated,
        # creators already lives in `raw`; break it out for convenient querying.
        creators=record.raw.get("attributes", {}).get("creators"),
        raw=record.raw,
        fetched_at=now,
    )
    return Write(groups=[[row]])
