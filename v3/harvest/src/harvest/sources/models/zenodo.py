"""Typed projection of a Zenodo record (one ``hits.hits[]`` entry).

Zenodo's ``GET /api/records?q=doi:"{doi}"`` returns a search envelope whose
``hits.hits`` array holds bare JSON objects — no schema, no validation. This
module is the schema it declines to give: frozen dataclasses over the fields
harvest actually reads, each with a tolerant ``from_dict``. ``from_result`` takes
one hit (``hits.hits[0]``); the ``ZenodoSource`` unwraps the envelope and decides
found vs not-found from ``hits.total``.

Design choices, mirroring ``OpenAlexWork`` / ``UnpaywallRecord`` / ``ArxivRecord``:

* **Tolerant, not strict.** Keep only the fields we know and ignore the rest —
  Zenodo (InvenioRDM legacy-compat serialization) adds/moves fields over time and
  a harvester should not break when it does. Nothing is lost regardless: the whole
  hit is retained on ``raw`` for the ``source_zenodo.raw`` jsonb column.
* **Everything optional.** Zenodo omits/nulls fields freely (``license`` is absent
  on closed records, ``files`` empty/absent on metadata-only records,
  ``version`` optional, …), so every field defaults and nested access is guarded.
* **Prefer top-level identity.** ``doi`` / ``conceptdoi`` are read from the
  top-level of the hit (``pids`` was null on the probed record); the version DOI
  lives in ``doi`` and the concept (all-versions) DOI in ``conceptdoi`` — they can
  differ from the queried DOI.

Field names/nesting verified against a live legacy Zenodo record response
(v3/harvest/tests/fixtures/zenodo_sample.json).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Creator:
    """One entry of ``metadata.creators`` — an author/contributor of the deposit."""

    name: str | None = None
    affiliation: str | None = None
    orcid: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Creator | None:
        if not data:
            return None
        return cls(
            name=data.get("name"),
            affiliation=data.get("affiliation"),
            orcid=data.get("orcid"),
        )


@dataclass(frozen=True)
class ZenodoFile:
    """One entry of the top-level ``files`` array — a deposited file."""

    key: str | None = None
    size: int | None = None
    checksum: str | None = None
    download_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ZenodoFile | None:
        if not data:
            return None
        links = data.get("links") or {}
        return cls(
            key=data.get("key"),
            size=data.get("size"),
            checksum=data.get("checksum"),
            download_url=links.get("self"),
        )


def _list(factory, data: list[dict[str, Any]] | None) -> list:
    """Map a list of dicts through a ``from_dict``, dropping anything that parses to None."""
    return [item for item in map(factory, data or []) if item is not None]


@dataclass(frozen=True)
class ZenodoRecord:
    """A typed Zenodo record — the subset harvest reads, plus ``raw`` for lossless persist.

    ``from_result`` takes one hit dict (``hits.hits[0]``). Keep the whole hit on
    ``raw`` so ``source_zenodo.raw`` stays a faithful copy even as this projection
    stays small.
    """

    record_id: int | None = None
    doi: str | None = None
    conceptdoi: str | None = None
    doi_url: str | None = None
    title: str | None = None
    publication_date: str | None = None
    description: str | None = None
    access_right: str | None = None
    version: str | None = None
    resource_type_type: str | None = None
    resource_type_title: str | None = None
    license_id: str | None = None
    creators: list[Creator] = field(default_factory=list)
    files: list[ZenodoFile] = field(default_factory=list)
    landing_page: str | None = None
    api_self: str | None = None
    created: str | None = None
    modified: str | None = None
    updated: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def file_download_url(self) -> str | None:
        """Direct content download URL of the first deposited file, if any."""
        return self.files[0].download_url if self.files else None

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> ZenodoRecord:
        metadata = result.get("metadata") or {}
        resource_type = metadata.get("resource_type") or {}
        license_ = metadata.get("license") or {}
        links = result.get("links") or {}
        return cls(
            record_id=result.get("id"),
            doi=result.get("doi"),
            conceptdoi=result.get("conceptdoi"),
            doi_url=result.get("doi_url"),
            # top-level title is duplicated at metadata.title; prefer top-level.
            title=result.get("title") or metadata.get("title"),
            publication_date=metadata.get("publication_date"),
            description=metadata.get("description"),
            access_right=metadata.get("access_right"),
            version=metadata.get("version"),
            resource_type_type=resource_type.get("type"),
            resource_type_title=resource_type.get("title"),
            license_id=license_.get("id"),
            creators=_list(Creator.from_dict, metadata.get("creators")),
            files=_list(ZenodoFile.from_dict, result.get("files")),
            landing_page=links.get("self_html"),
            api_self=links.get("self"),
            created=result.get("created"),
            modified=result.get("modified"),
            updated=result.get("updated"),
            raw=dict(result),
        )
