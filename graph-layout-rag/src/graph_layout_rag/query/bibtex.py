"""Deterministic BibTeX export for a manifest item (+ optional enrichment meta).

Field values stay UTF-8; only the cite KEY is ASCII-folded so it is a safe
LaTeX label. The entry type is inferred conservatively from venue/biblio signals:
``@inproceedings`` for conference-like venues, ``@article`` when a journal/venue
and year are present, else ``@misc``.
"""

from __future__ import annotations

import re
import unicodedata

from graph_layout_rag.citation_store import PaperMeta
from graph_layout_rag.manifest import ManifestItem, slug_id

_CONFERENCE_HINTS = (
    "conference",
    "proceedings",
    "symposium",
    "workshop",
    "proc.",
    "conf.",
)


def _ascii_fold(text: str) -> str:
    """NFKD-normalize and drop non-ASCII so a unicode name yields a plain key token."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "", folded)


def _first_author_surname(authors: list[str]) -> str | None:
    if not authors:
        return None
    name = authors[0].strip()
    if not name:
        return None
    # "Surname, Given" -> Surname; "Given Surname" -> last token.
    if "," in name:
        surname = name.split(",", 1)[0]
    else:
        surname = name.split()[-1]
    return surname


def _venue_from(item: ManifestItem, meta: PaperMeta | None) -> str | None:
    if item.venue:
        return item.venue
    if meta is not None and meta.biblio:
        for key in ("venue", "journal", "booktitle", "container_title"):
            value = meta.biblio.get(key)
            if value:
                return str(value)
    return None


def _authors_from(item: ManifestItem, meta: PaperMeta | None) -> list[str]:
    if item.authors:
        return [a for a in item.authors if a and a.strip()]
    if meta is not None and meta.full_authors:
        names: list[str] = []
        for entry in meta.full_authors:
            if isinstance(entry, str):
                name = entry
            elif isinstance(entry, dict):
                name = entry.get("name") or entry.get("display_name") or ""
            else:
                name = ""
            if name and name.strip():
                names.append(name.strip())
        return names
    return []


def _entry_type(venue: str | None, year: int | None) -> str:
    if venue:
        low = venue.lower()
        if any(hint in low for hint in _CONFERENCE_HINTS):
            return "inproceedings"
        if year is not None:
            return "article"
    return "misc"


def _cite_key(
    item: ManifestItem, authors: list[str], year: int | None, title: str | None
) -> str:
    surname = _first_author_surname(authors)
    surname_tok = _ascii_fold(surname) if surname else ""
    title_tok = ""
    if title:
        first_word = title.strip().split()
        if first_word:
            title_tok = _ascii_fold(first_word[0])
    year_tok = str(year) if year is not None else ""
    key = f"{surname_tok}{year_tok}{title_tok}".lower()
    if not key:
        key = slug_id(item.id) or "ref"
    return key


def bibtex_for_doc(item: ManifestItem, meta: PaperMeta | None = None) -> str:
    """Render a single BibTeX entry for a manifest item.

    Robust to missing authors/DOI/year/venue; unicode survives in field values
    while the cite key is ASCII-folded.
    """
    authors = _authors_from(item, meta)
    venue = _venue_from(item, meta)
    year = item.year
    title = item.title

    entry_type = _entry_type(venue, year)
    key = _cite_key(item, authors, year, title)

    fields: list[tuple[str, str]] = []
    if title:
        fields.append(("title", title))
    if authors:
        fields.append(("author", " and ".join(authors)))
    if year is not None:
        fields.append(("year", str(year)))
    if venue:
        venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
        fields.append((venue_field, venue))
    if item.doi:
        fields.append(("doi", item.doi))
    if item.url:
        fields.append(("url", item.url))

    lines = [f"@{entry_type}{{{key},"]
    for name, value in fields:
        lines.append(f"  {name} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)
