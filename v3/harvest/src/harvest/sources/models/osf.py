"""Typed projection of an OSF preprint (JSON:API detail response).

OSF's ``GET /v2/preprints/{id}/`` returns a JSON:API document — a
``{data: {id, attributes, relationships, links}, meta}`` envelope, no client
schema. This module is the schema it declines to give: frozen dataclasses over
the fields harvest actually reads, each with a tolerant ``from_dict``.

Design choices, mirroring ``UnpaywallRecord`` / ``OpenAlexWork`` / ``S2Record``:

* **Tolerant, not strict.** Keep only the fields we know and ignore the rest —
  OSF adds fields over time and a harvester should not break when it does.
  Nothing is lost regardless: the whole response is retained on ``raw`` for the
  ``source_osf.raw`` jsonb column.
* **Everything optional.** OSF omits/nulls fields freely (``attributes.doi`` is
  null on most preprints — the meaningful OSF DOI lives at
  ``links.preprint_doi`` — ``date_published`` is null until published, …), so
  every field defaults and nested access is guarded.

Field names/nesting typed from a live JSON:API detail response (meta.version
2.0). The whole payload is kept on ``raw`` so this projection can stay small.
See https://developer.osf.io/#operation/preprints_read for the full catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OsfLicenseRecord:
    """``attributes.license_record`` — the copyright terms attached to the preprint."""

    copyright_holders: list[str] = field(default_factory=list)
    year: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OsfLicenseRecord | None:
        if not data:
            return None
        return cls(
            copyright_holders=list(data.get("copyright_holders") or []),
            year=data.get("year"),
        )


def _rel_id(relationships: dict[str, Any], name: str) -> str | None:
    """The ``data.id`` of a JSON:API relationship (``relationships[name].data.id``)."""
    rel = relationships.get(name)
    if not isinstance(rel, dict):
        return None
    data = rel.get("data")
    if not isinstance(data, dict):
        return None
    return data.get("id")


def _rel_href(relationships: dict[str, Any], name: str) -> str | None:
    """The related-resource URL of a relationship (``relationships[name].links.related.href``)."""
    rel = relationships.get(name)
    if not isinstance(rel, dict):
        return None
    links = rel.get("links")
    if not isinstance(links, dict):
        return None
    related = links.get("related")
    if not isinstance(related, dict):
        return None
    return related.get("href")


@dataclass(frozen=True)
class OsfPreprint:
    """A typed OSF preprint — the subset harvest reads, plus ``raw`` for lossless persist.

    ``from_result`` takes the whole JSON:API response dict the API hands back for
    one preprint (the ``{data, meta}`` envelope). Keep it on ``raw`` so
    ``source_osf.raw`` stays a faithful copy even as this projection stays small.
    """

    osf_id: str
    title: str
    doi: str | None = None
    description: str | None = None
    date_published: str | None = None
    date_created: str | None = None
    date_modified: str | None = None
    is_published: bool | None = None
    reviews_state: str | None = None
    tags: list[str] = field(default_factory=list)
    subjects: list[list[dict[str, Any]]] = field(default_factory=list)
    license_record: OsfLicenseRecord | None = None
    preprint_doi: str | None = None
    html_url: str | None = None
    self_url: str | None = None
    primary_file_id: str | None = None
    primary_file_url: str | None = None
    files_url: str | None = None
    provider_id: str | None = None
    node_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> OsfPreprint:
        data = result.get("data")
        if not isinstance(data, dict):
            data = {}
        attributes = data.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        links = data.get("links")
        if not isinstance(links, dict):
            links = {}
        relationships = data.get("relationships")
        if not isinstance(relationships, dict):
            relationships = {}

        return cls(
            osf_id=data.get("id") or "",
            title=attributes.get("title") or "",
            doi=attributes.get("doi"),
            description=attributes.get("description"),
            date_published=attributes.get("date_published"),
            date_created=attributes.get("date_created"),
            date_modified=attributes.get("date_modified"),
            is_published=attributes.get("is_published"),
            reviews_state=attributes.get("reviews_state"),
            tags=list(attributes.get("tags") or []),
            subjects=list(attributes.get("subjects") or []),
            license_record=OsfLicenseRecord.from_dict(attributes.get("license_record")),
            preprint_doi=links.get("preprint_doi"),
            html_url=links.get("html"),
            self_url=links.get("self"),
            primary_file_id=_rel_id(relationships, "primary_file"),
            primary_file_url=_rel_href(relationships, "primary_file"),
            files_url=_rel_href(relationships, "files"),
            provider_id=_rel_id(relationships, "provider"),
            node_id=_rel_id(relationships, "node"),
            raw=dict(result),
        )
