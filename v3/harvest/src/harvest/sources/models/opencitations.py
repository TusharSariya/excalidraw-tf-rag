"""Typed projection of an OpenCitations Index v2 citation record.

OpenCitations' ``GET /index/v2/references/{id}`` and ``GET /index/v2/citations/{id}``
each return a bare JSON array of citation objects — no schema, no validation. This
module is the schema they decline to give: frozen dataclasses over the fields
harvest reads, each with a tolerant ``from_dict``.

Design choices, mirroring ``OpenAlexWork`` / ``UnpaywallRecord`` / ``S2Record``:

* **Tolerant, not strict.** Keep only the fields we know and ignore the rest — a
  harvester should not break when OpenCitations adds a field. Nothing is lost
  regardless: each element's whole dict is retained on ``OpenCitation.raw`` (the
  home of the untyped ``citing``/``cited`` bundle ids), so
  ``opencitations_edge.raw`` stays a faithful copy of the response.
* **Everything optional.** OpenCitations omits/partial-fills fields freely (a
  ``citing``/``cited`` bundle may carry no ``doi:`` token; ``creation`` may be a
  bare year; ``timespan`` may be negative), so every field defaults and parsing
  is guarded.

The one non-trivial shape: ``citing`` and ``cited`` are space-separated identifier
BUNDLES (e.g. ``'omid:br/062 doi:10.1/x openalex:W1 pmid:2'``), NOT bare DOIs.
``citing_doi`` / ``cited_doi`` are PARSED out of the ``doi:`` token (DOIs are
returned lowercased) and may be absent when the bundle carries only omid/openalex/
isbn ids.
See https://opencitations.net/index/api/v2 for the endpoint catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _doi_from_bundle(bundle: str | None) -> str | None:
    """Extract the bare DOI from a space-separated OpenCitations id bundle.

    A bundle looks like ``'omid:br/062 doi:10.1/x openalex:W1 pmid:2'``. Pull the
    token prefixed ``doi:`` and return its bare value (already lowercased
    upstream); return None when the bundle carries no ``doi:`` token.
    """
    if not bundle:
        return None
    for token in bundle.split():
        if token.startswith("doi:"):
            return token[len("doi:") :] or None
    return None


def _yn(value: str | None) -> bool | None:
    """Map OpenCitations ``'yes'``/``'no'`` self-citation flags to bool; None otherwise."""
    if value is None:
        return None
    low = value.strip().lower()
    if low == "yes":
        return True
    if low == "no":
        return False
    return None


@dataclass(frozen=True)
class OpenCitation:
    """One citation edge element from a ``references/`` or ``citations/`` response.

    ``citing_doi`` / ``cited_doi`` are parsed out of the ``citing`` / ``cited``
    bundles; either may be None. The whole element dict is kept on ``raw`` so the
    untyped bundle ids (omid/openalex/pmid) survive on ``opencitations_edge.raw``.
    """

    oci: str | None = None
    citing: str | None = None
    cited: str | None = None
    citing_doi: str | None = None
    cited_doi: str | None = None
    creation: str | None = None
    timespan: str | None = None
    journal_sc: str | None = None
    author_sc: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def journal_self_citation(self) -> bool | None:
        """``journal_sc`` as a bool ('yes'->True, 'no'->False)."""
        return _yn(self.journal_sc)

    @property
    def author_self_citation(self) -> bool | None:
        """``author_sc`` as a bool ('yes'->True, 'no'->False)."""
        return _yn(self.author_sc)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OpenCitation | None:
        if not data:
            return None
        return cls(
            oci=data.get("oci"),
            citing=data.get("citing"),
            cited=data.get("cited"),
            citing_doi=_doi_from_bundle(data.get("citing")),
            cited_doi=_doi_from_bundle(data.get("cited")),
            creation=data.get("creation"),
            timespan=data.get("timespan"),
            journal_sc=data.get("journal_sc"),
            author_sc=data.get("author_sc"),
            raw=dict(data),
        )


def _list(data: list[dict[str, Any]] | None) -> list[OpenCitation]:
    """Map a list of dicts through ``from_dict``, dropping anything that parses to None."""
    return [item for item in map(OpenCitation.from_dict, data or []) if item is not None]


@dataclass(frozen=True)
class OpenCitationsRecord:
    """Both directions of the OpenCitations index for one seed DOI.

    ``references`` are the works the seed CITES (from ``/references/doi:{seed}``,
    where every element is ``citing=seed, cited=other``). ``citations`` are the
    works that CITE the seed (from ``/citations/doi:{seed}``, where every element
    is ``citing=other, cited=seed``). The full arrays are kept on ``raw``.
    """

    doi: str
    references: list[OpenCitation] = field(default_factory=list)
    citations: list[OpenCitation] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_responses(
        cls,
        doi: str,
        references: list[dict[str, Any]] | None,
        citations: list[dict[str, Any]] | None,
    ) -> OpenCitationsRecord:
        return cls(
            doi=doi,
            references=_list(references),
            citations=_list(citations),
            raw={"references": references or [], "citations": citations or []},
        )
