"""Typed projection of a DataCite DOI record.

The DataCite REST API returns a JSON:API envelope — ``{"data": {"id", "type",
"attributes": {…}}}`` for a single DOI, ``{"data": [ …same shape… ]}`` for a
query. This module is the schema the raw envelope declines to give: frozen
dataclasses over the subset harvest reads, each with a tolerant ``from_dict``.

Design choices, mirroring ``OpenAlexWork`` / ``ArxivRecord`` / ``S2Record``:

* **Tolerant, not strict.** Keep only the fields we know and ignore the rest —
  DataCite carries dozens of attributes (``geoLocations``, ``fundingReferences``,
  ``viewsOverTime``, …) we do not project. Nothing is lost regardless: the whole
  record is retained on ``raw`` for the ``source_datacite.raw`` jsonb column.
* **Everything optional.** DataCite omits/nulls fields freely (``language`` and
  ``version`` are null in the sample, ``subjects``/``descriptions`` are often
  empty lists), so every field defaults and nested access is guarded.

``from_result`` takes ONE record's envelope item — the ``{"id", "type",
"attributes"}`` object — not the outer ``{"data": …}`` wrapper. Field
names/nesting verified against a live single-DOI response (the zenodo fixture in
``tests/fixtures/datacite_sample.json``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Title:
    """A ``titles[]`` item. Primary title text lives at ``titles[0].title``."""

    title: str | None = None
    title_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Title | None:
        if not data:
            return None
        return cls(title=data.get("title"), title_type=data.get("titleType"))


@dataclass(frozen=True)
class Creator:
    """A ``creators[]`` item — an author/contributor of the resource."""

    name: str | None = None
    name_type: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    affiliation: list[str] = field(default_factory=list)
    name_identifiers: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Creator | None:
        if not data:
            return None
        return cls(
            name=data.get("name"),
            name_type=data.get("nameType"),
            given_name=data.get("givenName"),
            family_name=data.get("familyName"),
            affiliation=list(data.get("affiliation") or []),
            name_identifiers=list(data.get("nameIdentifiers") or []),
        )


@dataclass(frozen=True)
class Description:
    """A ``descriptions[]`` item — abstract-like text."""

    description: str | None = None
    description_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Description | None:
        if not data:
            return None
        return cls(
            description=data.get("description"),
            description_type=data.get("descriptionType"),
        )


@dataclass(frozen=True)
class Types:
    """The ``types`` block — one resource classified across several vocabularies."""

    ris: str | None = None
    bibtex: str | None = None
    citeproc: str | None = None
    schema_org: str | None = None
    resource_type: str | None = None
    resource_type_general: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Types | None:
        if not data:
            return None
        return cls(
            ris=data.get("ris"),
            bibtex=data.get("bibtex"),
            citeproc=data.get("citeproc"),
            schema_org=data.get("schemaOrg"),
            resource_type=data.get("resourceType"),
            resource_type_general=data.get("resourceTypeGeneral"),
        )


@dataclass(frozen=True)
class Rights:
    """A ``rightsList[]`` item — a license statement."""

    rights: str | None = None
    rights_uri: str | None = None
    scheme_uri: str | None = None
    rights_identifier: str | None = None
    rights_identifier_scheme: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Rights | None:
        if not data:
            return None
        return cls(
            rights=data.get("rights"),
            rights_uri=data.get("rightsUri"),
            scheme_uri=data.get("schemeUri"),
            rights_identifier=data.get("rightsIdentifier"),
            rights_identifier_scheme=data.get("rightsIdentifierScheme"),
        )


@dataclass(frozen=True)
class DateInfo:
    """A ``dates[]`` item — a typed date (Issued / Updated / …)."""

    date: str | None = None
    date_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DateInfo | None:
        if not data:
            return None
        return cls(date=data.get("date"), date_type=data.get("dateType"))


@dataclass(frozen=True)
class Subject:
    """A ``subjects[]`` item — a keyword/subject term (empty list in the sample)."""

    subject: str | None = None
    subject_scheme: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Subject | None:
        if not data:
            return None
        return cls(subject=data.get("subject"), subject_scheme=data.get("subjectScheme"))


@dataclass(frozen=True)
class RelatedIdentifier:
    """A ``relatedIdentifiers[]`` item — a typed edge to another artefact."""

    relation_type: str | None = None
    related_identifier: str | None = None
    related_identifier_type: str | None = None
    resource_type_general: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RelatedIdentifier | None:
        if not data:
            return None
        return cls(
            relation_type=data.get("relationType"),
            related_identifier=data.get("relatedIdentifier"),
            related_identifier_type=data.get("relatedIdentifierType"),
            resource_type_general=data.get("resourceTypeGeneral"),
        )


def _list(factory, data: list[dict[str, Any]] | None) -> list:
    """Map a list of dicts through a ``from_dict``, dropping anything that parses to None."""
    return [item for item in map(factory, data or []) if item is not None]


@dataclass(frozen=True)
class DataCiteRecord:
    """A typed DataCite DOI — the subset harvest reads, plus ``raw`` for lossless persist.

    ``from_result`` takes one JSON:API record item (``{"id", "type",
    "attributes"}``). The whole item is kept on ``raw`` so ``source_datacite.raw``
    stays a faithful copy even as this projection stays small.
    """

    doi: str
    record_type: str | None = None
    prefix: str | None = None
    suffix: str | None = None
    state: str | None = None
    titles: list[Title] = field(default_factory=list)
    creators: list[Creator] = field(default_factory=list)
    publisher: str | None = None
    publication_year: int | None = None
    types: Types | None = None
    descriptions: list[Description] = field(default_factory=list)
    subjects: list[Subject] = field(default_factory=list)
    dates: list[DateInfo] = field(default_factory=list)
    url: str | None = None
    content_url: str | None = None
    version: str | None = None
    language: str | None = None
    rights_list: list[Rights] = field(default_factory=list)
    related_identifiers: list[RelatedIdentifier] = field(default_factory=list)
    schema_version: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    created: str | None = None
    registered: str | None = None
    updated: str | None = None
    is_active: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str | None:
        """The primary title text (``titles[0].title``), if any."""
        return self.titles[0].title if self.titles else None

    @property
    def resource_type_general(self) -> str | None:
        """``types.resourceTypeGeneral`` — 'Software' / 'Dataset' / …."""
        return self.types.resource_type_general if self.types else None

    @property
    def resource_type(self) -> str | None:
        """``types.resourceType`` — the free-text subtype (may be empty)."""
        return self.types.resource_type if self.types else None

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> DataCiteRecord:
        """Build from one JSON:API record item — ``{"id", "type", "attributes"}``."""
        attrs = result.get("attributes") or {}
        return cls(
            # attributes.doi is the canonical DOI; data.id equals it. Prefer the
            # attribute, fall back to the envelope id.
            doi=attrs.get("doi") or result.get("id") or "",
            record_type=result.get("type"),
            prefix=attrs.get("prefix"),
            suffix=attrs.get("suffix"),
            state=attrs.get("state"),
            titles=_list(Title.from_dict, attrs.get("titles")),
            creators=_list(Creator.from_dict, attrs.get("creators")),
            publisher=attrs.get("publisher"),
            publication_year=attrs.get("publicationYear"),
            types=Types.from_dict(attrs.get("types")),
            descriptions=_list(Description.from_dict, attrs.get("descriptions")),
            subjects=_list(Subject.from_dict, attrs.get("subjects")),
            dates=_list(DateInfo.from_dict, attrs.get("dates")),
            url=attrs.get("url"),
            content_url=attrs.get("contentUrl"),
            version=attrs.get("version"),
            language=attrs.get("language"),
            rights_list=_list(Rights.from_dict, attrs.get("rightsList")),
            related_identifiers=_list(RelatedIdentifier.from_dict, attrs.get("relatedIdentifiers")),
            schema_version=attrs.get("schemaVersion"),
            citation_count=attrs.get("citationCount"),
            reference_count=attrs.get("referenceCount"),
            created=attrs.get("created"),
            registered=attrs.get("registered"),
            updated=attrs.get("updated"),
            is_active=attrs.get("isActive"),
            # Keep the whole JSON:API item so the `raw` jsonb column is lossless.
            raw=dict(result),
        )
