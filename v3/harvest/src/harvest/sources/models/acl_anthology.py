"""Typed projection of an ACL Anthology BibTeX record.

ACL Anthology exposes no per-paper JSON API. The per-record surface is the BibTeX
file ``GET https://aclanthology.org/{anthology_id}.bib``, and the native batch
surface is the full-corpus dump ``https://aclanthology.org/anthology.bib.gz`` — one
gzipped file of every record. Either way the payload is BibTeX text, not JSON, so
``sources.acl_anthology`` parses each ``@entry{…}`` block into a plain dict and this
module types the subset harvest reads over it.

Design choices, mirroring ``OpenAlexWork`` / ``UnpaywallRecord`` / ``ArxivRecord``:

* **Tolerant, not strict.** Keep only the fields we know and ignore the rest —
  BibTeX carries many optional keys (editor, publisher, address, month, pages) and a
  harvester should not break when one is missing. Nothing is lost regardless: the
  whole parsed entry (plus the real ``raw_bibtex`` bytes) is retained on ``raw`` for
  the ``source_acl_anthology.raw`` jsonb column.
* **Everything optional.** Only ``anthology_id`` (the request key) is guaranteed; a
  record may lack a ``doi``, and ``venue`` comes from ``booktitle`` (inproceedings) or
  ``journal`` (article), so every field defaults and access is guarded.

Parsing watch-outs baked into ``from_dict``:

* ``title`` / ``venue`` carry TeX brace protection (``{BERT}``, ``{A}merican``) that is
  typographic, not data — the braces are stripped.
* ``author`` / ``editor`` are a single ``' and '``-joined string of ``Last, First``
  names, split back into a list.
* ``year`` is a quoted string in the bib; coerced to ``int``.
* ``anthology_id`` is the request key, not a bib field — it is recovered from the
  ``url`` path by the source and handed in on the parsed dict.
* ``pdf_url`` has no explicit bib field — it is derived from ``url`` (landing page
  ``…/N19-1423/`` -> ``…/N19-1423.pdf``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# BibTeX values wrap across lines and pad with runs of spaces; collapse to single
# spaces so multi-line titles / ' and '-joined author strings normalise cleanly.
_WS = re.compile(r"\s+")

# ACL bib joins names with ' and ' (surrounded by the wrapped whitespace above,
# already collapsed to single spaces by the time this runs).
_AND = re.compile(r"\s+and\s+")

# A 4-digit run anywhere in the year value ("2019", "2019.", ...).
_YEAR = re.compile(r"\d{4}")


def _clean(value: str | None) -> str | None:
    """Strip TeX brace protection and surrounding whitespace; None on empty.

    ACL protects case/acronyms with braces (``{BERT}``, ``{A}merican``). They are
    typographic, carry no data, and must not survive into typed text.
    """
    if value is None:
        return None
    cleaned = _WS.sub(" ", value.replace("{", "").replace("}", "")).strip()
    return cleaned or None


def _split_names(value: str | None) -> list[str]:
    """``'Last, First and Last, First'`` -> ``['Last, First', 'Last, First']``.

    The bib ``author`` / ``editor`` field is one ``' and '``-joined string, not a
    list. Braces are stripped per-name (some names carry ``{van der Berg}``).
    """
    if not value:
        return []
    collapsed = _WS.sub(" ", value).strip()
    if not collapsed:
        return []
    return [name for part in _AND.split(collapsed) if (name := _clean(part))]


def _coerce_year(value: str | None) -> int | None:
    """Quoted bib year string -> int; None if no 4-digit run is present."""
    if value is None:
        return None
    m = _YEAR.search(str(value))
    return int(m.group()) if m else None


def _derive_pdf_url(url: str | None) -> str | None:
    """Landing page ``…/N19-1423/`` -> PDF ``…/N19-1423.pdf`` (verified live 200)."""
    if not url:
        return None
    return f"{url.rstrip('/')}.pdf"


@dataclass(frozen=True)
class AclAnthologyRecord:
    """A typed ACL Anthology record — the subset harvest reads, plus ``raw``.

    ``from_dict`` takes the parsed BibTeX entry the source hands back (fields under
    ``bibtex``, the request key under ``anthology_id``, the real bytes under
    ``raw_bibtex``). Keep the whole dict on ``raw`` so ``source_acl_anthology.raw``
    stays a faithful copy even as this projection stays small.
    """

    anthology_id: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    editors: list[str] = field(default_factory=list)
    publisher: str | None = None
    address: str | None = None
    month: str | None = None
    pages: str | None = None
    bibkey: str | None = None
    entry_type: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AclAnthologyRecord:
        bib = data.get("bibtex") or {}
        url = bib.get("url")
        # inproceedings -> booktitle; @article -> journal.
        venue = bib.get("booktitle") or bib.get("journal")
        return cls(
            anthology_id=data.get("anthology_id") or "",
            title=_clean(bib.get("title")),
            authors=_split_names(bib.get("author")),
            year=_coerce_year(bib.get("year")),
            venue=_clean(venue),
            doi=bib.get("doi"),
            url=url,
            pdf_url=_derive_pdf_url(url),
            editors=_split_names(bib.get("editor")),
            publisher=_clean(bib.get("publisher")),
            address=_clean(bib.get("address")),
            month=bib.get("month"),
            pages=bib.get("pages"),
            bibkey=data.get("bibkey"),
            entry_type=data.get("entry_type"),
            raw=dict(data),
        )
