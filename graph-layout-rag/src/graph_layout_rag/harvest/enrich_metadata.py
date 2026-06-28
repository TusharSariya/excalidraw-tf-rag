"""Resumable metadata-enrichment backfill for the corpus.

For every ingested manifest item (status ``ok`` / ``metadata_only``) this pass fans out to
three providers and folds the results into (a) new manifest fields and (b) the
``papers_meta`` enrichment table in ``citations.sqlite``:

  * **OpenAlex** (by DOI, with id / arXiv / title-search fallbacks): fwci, cited-by count,
    retraction flag, reconstructed abstract, venue + venue type + OA version + genre,
    best-OA pdf/license, biblio block, full author list.
  * **Semantic Scholar** ``/paper/batch``: tldr, publicationTypes (genre fallback),
    open-access pdf (fallback), and the SPECTER2 embedding (stored in ``doc_specter2``).
  * **arXiv** Atom API: subject categories → ``arxiv_category`` (+ discovered arXiv id).

Resumability (the interrupt-safe ordering):
  Per batch we save the manifest **before** writing ``papers_meta``. Because ``papers_meta``
  is the resume marker (``has_paper_meta``), any row that exists already had its manifest
  fields persisted, so a crash only re-does the in-flight batch (idempotent). An item where
  every provider cleanly missed still gets a ``source_provider="none"`` marker so it is not
  retried; an item with a *transient* provider failure (rate-limit / timeout / 5xx) is left
  unmarked and picked up next run.

The CLI is a thin wrapper in ``harvest/run.py`` (``harvest enrich-metadata``). Network is
never exercised in tests — the provider entry points are monkeypatched.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable

import httpx

from graph_layout_rag import citation_store as cs
from graph_layout_rag.harvest.arxiv import ARXIV_API, ATOM_NS, _arxiv_id_from_entry
from graph_layout_rag.harvest.log import get_logger
from graph_layout_rag.harvest.openalex import OPENALEX_API, _abstract_from_inverted_index
from graph_layout_rag.harvest.parallel import parallel_map
from graph_layout_rag.harvest.providers import (
    OPENALEX,
    SEMANTIC_SCHOLAR,
    OutcomeKind,
)
from graph_layout_rag.manifest import Manifest, ManifestItem, load_manifest, save_manifest

S2_BATCH_API = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "tldr,publicationTypes,openAccessPdf,embedding.specter_v2,externalIds"
ARXIV_NS = {"arxiv": "http://arxiv.org/schemas/atom"}

_MAILTO = "graph-layout-rag@excalidraw-tf.local"
_INGESTED_STATUSES = frozenset({"ok", "metadata_only"})
_DEFAULT_BATCH = 200
_S2_BATCH = 400
_ARXIV_BATCH = 50


# --------------------------------------------------------------------------- helpers

def _chunks(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _is_synthetic_arxiv_doi(doi: str | None) -> bool:
    """A DOI auto-minted from an arXiv id (``10.48550/arXiv.NNNN``). These don't always
    resolve in OpenAlex, so callers prefer the arXiv-id / title-search fallbacks."""
    return bool(doi) and "arxiv" in (doi or "").lower()


def _ext_get(item: ManifestItem, *keys: str) -> str | None:
    ext = item.externalIds or {}
    lowered = {k.lower(): v for k, v in ext.items()}
    for key in keys:
        val = lowered.get(key.lower())
        if val:
            return val
    return None


def _arxiv_id_for_item(item: ManifestItem) -> str | None:
    """Best-effort arXiv id from externalIds, a synthetic arXiv DOI, or the URL."""
    explicit = _ext_get(item, "ArXiv", "arxiv")
    if explicit:
        return explicit
    doi = item.doi or ""
    m = re.search(r"arxiv\.([\w.]+\d)", doi, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:\.pdf)?$", item.url or "", re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _arxiv_base(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


# --------------------------------------------------------------------------- OpenAlex

@dataclass
class _ProviderHit:
    """One provider's contribution for an item: ``data`` is None on a clean miss;
    ``transient`` flags a retryable failure (rate-limit / timeout / 5xx)."""
    data: dict | None = None
    transient: bool = False


def _extract_openalex(work: dict) -> dict:
    prim = work.get("primary_location") or {}
    src = prim.get("source") or {}
    boa = work.get("best_oa_location") or {}
    authors = [
        (a.get("author") or {}).get("display_name")
        for a in work.get("authorships") or []
        if (a.get("author") or {}).get("display_name")
    ]
    return {
        "fwci": work.get("fwci"),
        "cited_by_count": work.get("cited_by_count"),
        "is_retracted": bool(work.get("is_retracted")),
        "abstract": _abstract_from_inverted_index(work.get("abstract_inverted_index")),
        "venue": src.get("display_name"),
        "venue_type": src.get("type"),
        "oa_version": prim.get("version"),
        "genre": work.get("type"),
        "oa_pdf_url": boa.get("pdf_url") or boa.get("landing_page_url"),
        "license": boa.get("license"),
        "biblio": work.get("biblio") or None,
        "full_authors": authors or None,
    }


def _oa_lookups(item: ManifestItem) -> list[tuple[str, str]]:
    """Ordered (kind, value) lookups to try until one resolves."""
    out: list[tuple[str, str]] = []
    doi = cs.normalize_doi(item.doi)
    if doi and not _is_synthetic_arxiv_doi(item.doi):
        out.append(("doi", doi))
    oaid = cs.normalize_oa_id(_ext_get(item, "OpenAlex", "openalex"))
    if oaid:
        out.append(("oaid", oaid))
    arxiv_id = _arxiv_id_for_item(item)
    if arxiv_id:
        out.append(("doi", f"10.48550/arxiv.{_arxiv_base(arxiv_id)}"))
    if item.title:
        out.append(("search", item.title))
    return out


def _oa_request(kind: str, value: str):
    if kind == "doi":
        # Canonical OpenAlex DOI lookup form. The full-URL form
        # (/works/https://doi.org/<doi>) is rejected with 429 (the embedded
        # scheme trips OpenAlex's path handling), which under concurrency
        # starves the shared rate budget and breaks the title-search fallback
        # too — so DOI-bearing items resolved at ~4% vs ~92% for no-DOI items.
        url = f"{OPENALEX_API}/doi:{value}"
        params = {"mailto": _MAILTO}
    elif kind == "oaid":
        url = f"{OPENALEX_API}/{value}"
        params = {"mailto": _MAILTO}
    else:  # search
        url = OPENALEX_API
        params = {"search": value, "per_page": "1", "mailto": _MAILTO}
    return OPENALEX.request("GET", url, params=params, timeout=30.0)


def _fetch_openalex(item: ManifestItem) -> _ProviderHit:
    transient = False
    for kind, value in _oa_lookups(item):
        outcome = _oa_request(kind, value)
        if outcome.kind is OutcomeKind.SUCCESS and outcome.data:
            work = outcome.data
            if kind == "search":
                results = (work or {}).get("results") or []
                if not results:
                    continue
                work = results[0]
            return _ProviderHit(data=_extract_openalex(work))
        if outcome.kind not in (OutcomeKind.SUCCESS, OutcomeKind.TERMINAL_MISS):
            transient = True
    return _ProviderHit(data=None, transient=transient)


# --------------------------------------------------------------------------- Sem. Scholar

def _s2_id_for_item(item: ManifestItem) -> str | None:
    doi = cs.normalize_doi(item.doi)
    arxiv_id = _arxiv_id_for_item(item)
    if doi and not _is_synthetic_arxiv_doi(item.doi):
        return f"DOI:{doi}"
    if arxiv_id:
        return f"ARXIV:{_arxiv_base(arxiv_id)}"
    if doi:
        return f"DOI:{doi}"
    return None


def _extract_s2(paper: dict) -> dict:
    tldr = (paper.get("tldr") or {}).get("text")
    pub_types = paper.get("publicationTypes") or []
    embedding = paper.get("embedding") or {}
    return {
        "tldr": tldr,
        "genre": "/".join(pub_types) if pub_types else None,
        "oa_pdf_url": (paper.get("openAccessPdf") or {}).get("url"),
        "specter2": embedding.get("vector") or None,
        "arxiv_id": (paper.get("externalIds") or {}).get("ArXiv"),
    }


def _fetch_s2_batch(items: list[ManifestItem]) -> dict[str, _ProviderHit]:
    """Map doc_id -> hit for the S2 paper/batch endpoint. Items sharing an S2 id all get the
    paper's data. A transient batch failure marks every item in that chunk transient."""
    out: dict[str, _ProviderHit] = {}
    by_s2: list[tuple[str, ManifestItem]] = []
    for item in items:
        sid = _s2_id_for_item(item)
        if sid:
            by_s2.append((sid, item))
        else:
            out[item.id] = _ProviderHit(data=None)

    for chunk in _chunks(by_s2, _S2_BATCH):
        ids = [sid for sid, _ in chunk]
        outcome = SEMANTIC_SCHOLAR.request(
            "POST",
            S2_BATCH_API,
            params={"fields": S2_FIELDS},
            json={"ids": ids},
            timeout=90.0,
        )
        if outcome.kind is not OutcomeKind.SUCCESS:
            transient = outcome.kind is not OutcomeKind.TERMINAL_MISS
            for _, item in chunk:
                out[item.id] = _ProviderHit(data=None, transient=transient)
            continue
        papers = outcome.data or []
        for (_, item), paper in zip(chunk, papers):
            out[item.id] = _ProviderHit(data=_extract_s2(paper) if paper else None)
    return out


# --------------------------------------------------------------------------- arXiv

def _arxiv_query(ids: list[str]) -> str | None:
    """Raw Atom XML for an arXiv ``id_list`` (monkeypatched in tests)."""
    params = {"id_list": ",".join(ids), "max_results": str(len(ids))}
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            res = client.get(ARXIV_API, params=params)
            res.raise_for_status()
            return res.text
    except Exception:
        return None


def _extract_arxiv_entry(entry: ET.Element) -> dict:
    cats = [
        c.attrib["term"]
        for c in entry.findall("atom:category", ATOM_NS)
        if c.attrib.get("term")
    ]
    primary_el = entry.find("arxiv:primary_category", ARXIV_NS)
    primary = primary_el.attrib.get("term") if primary_el is not None else None
    ordered: list[str] = []
    for term in ([primary] if primary else []) + cats:
        if term and term not in ordered:
            ordered.append(term)
    return {
        "arxiv_category": ",".join(ordered) if ordered else None,
        "arxiv_id": _arxiv_id_from_entry(entry),
    }


def _fetch_arxiv(items: list[ManifestItem]) -> dict[str, _ProviderHit]:
    out: dict[str, _ProviderHit] = {}
    targets: list[tuple[str, ManifestItem]] = []
    for item in items:
        aid = _arxiv_id_for_item(item)
        if aid:
            targets.append((_arxiv_base(aid), item))
        else:
            out[item.id] = _ProviderHit(data=None)

    for chunk in _chunks(targets, _ARXIV_BATCH):
        ids = [aid for aid, _ in chunk]
        xml = _arxiv_query(ids)
        if not xml:
            for _, item in chunk:
                out[item.id] = _ProviderHit(data=None, transient=True)
            continue
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            for _, item in chunk:
                out[item.id] = _ProviderHit(data=None, transient=True)
            continue
        by_id: dict[str, dict] = {}
        for entry in root.findall("atom:entry", ATOM_NS):
            data = _extract_arxiv_entry(entry)
            if data.get("arxiv_id"):
                by_id[_arxiv_base(data["arxiv_id"])] = data
        for aid, item in chunk:
            data = by_id.get(aid)
            out[item.id] = _ProviderHit(data=data) if data else _ProviderHit(data=None)
    return out


# --------------------------------------------------------------------------- orchestration

def _apply_to_manifest(item: ManifestItem, oa: dict | None, s2: dict | None, ax: dict | None) -> bool:
    """Fold provider data into the item's new fields. Returns True if a missing abstract was
    backfilled."""
    oa = oa or {}
    s2 = s2 or {}
    ax = ax or {}

    if oa.get("is_retracted"):
        item.is_retracted = True
    if not item.venue and oa.get("venue"):
        item.venue = oa["venue"]
    if not item.venue_type and oa.get("venue_type"):
        item.venue_type = oa["venue_type"]
    if not item.oa_version and oa.get("oa_version"):
        item.oa_version = oa["oa_version"]
    if not item.genre:
        item.genre = oa.get("genre") or s2.get("genre")
    if not item.arxiv_category and ax.get("arxiv_category"):
        item.arxiv_category = ax["arxiv_category"]

    arxiv_id = ax.get("arxiv_id") or s2.get("arxiv_id")
    if arxiv_id and not _ext_get(item, "ArXiv", "arxiv"):
        item.externalIds = {**(item.externalIds or {}), "ArXiv": arxiv_id}

    backfilled = False
    if not (item.abstract and item.abstract.strip()):
        abstract = oa.get("abstract")
        if abstract and abstract.strip():
            item.abstract = abstract.strip()
            item.abstractSource = "openalex"
            backfilled = True
    return backfilled


def _resolved_abstract(item: ManifestItem, oa: dict | None) -> str | None:
    if item.abstract and item.abstract.strip():
        return item.abstract.strip()
    abstract = (oa or {}).get("abstract")
    return abstract.strip() if abstract and abstract.strip() else None


def _process_batch(
    db,
    manifest: Manifest,
    batch: list[ManifestItem],
    *,
    workers: int,
    dry_run: bool,
    stats: dict[str, int],
    log,
) -> None:
    oa_hits = parallel_map(_fetch_openalex, batch, workers=workers, label="enrich-metadata oa")
    oa_by_id = {item.id: hit for item, hit in zip(batch, oa_hits)}
    s2_by_id = _fetch_s2_batch(batch)
    ax_by_id = _fetch_arxiv(batch)

    # 1. Update in-memory manifest fields, then save FIRST (resume ordering).
    for item in batch:
        oa = oa_by_id.get(item.id, _ProviderHit()).data
        s2 = s2_by_id.get(item.id, _ProviderHit()).data
        ax = ax_by_id.get(item.id, _ProviderHit()).data
        if _apply_to_manifest(item, oa, s2, ax):
            stats["abstracts_backfilled"] += 1

    if not dry_run:
        save_manifest(manifest)

    # 2. THEN upsert papers_meta (+ specter2), or a terminal marker, or defer.
    for item in batch:
        oa_hit = oa_by_id.get(item.id, _ProviderHit())
        s2_hit = s2_by_id.get(item.id, _ProviderHit())
        ax_hit = ax_by_id.get(item.id, _ProviderHit())
        oa, s2, ax = oa_hit.data, s2_hit.data, ax_hit.data
        got_data = any(d is not None for d in (oa, s2, ax))
        had_transient = oa_hit.transient or s2_hit.transient or ax_hit.transient

        if not got_data:
            if had_transient:
                stats["transient_deferred"] += 1
                continue
            stats["terminal_miss"] += 1
            if not dry_run:
                cs.upsert_paper_meta(db, item.id, source_provider="none")
            continue

        providers = [name for name, d in (("openalex", oa), ("s2", s2), ("arxiv", ax)) if d]
        oa = oa or {}
        s2 = s2 or {}
        if not dry_run:
            cs.upsert_paper_meta(
                db,
                item.id,
                tldr=s2.get("tldr"),
                abstract=_resolved_abstract(item, oa),
                fwci=oa.get("fwci"),
                cited_by_count=oa.get("cited_by_count"),
                oa_pdf_url=oa.get("oa_pdf_url") or s2.get("oa_pdf_url"),
                license=oa.get("license"),
                biblio=oa.get("biblio"),
                full_authors=oa.get("full_authors"),
                source_provider="+".join(providers),
            )
            vec = s2.get("specter2")
            if vec:
                cs.upsert_specter2(db, item.id, list(vec))
        if s2.get("specter2"):
            stats["specter2_stored"] += 1
        stats["enriched"] += 1
    log.info("enrich-metadata: batch of %d done; running totals=%s", len(batch), stats)


def enrich_metadata(
    *,
    workers: int = 16,
    force: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    batch_size: int = _DEFAULT_BATCH,
) -> dict[str, int]:
    """Backfill provider metadata into the manifest + ``papers_meta``. Resumable: re-running
    skips items that already have a ``papers_meta`` row (unless ``force``)."""
    log = get_logger()
    manifest = load_manifest()
    db = cs.connect(cs.CITATIONS_DB_PATH)

    items = [i for i in manifest.items if i.status in _INGESTED_STATUSES]
    if not force:
        items = [i for i in items if not cs.has_paper_meta(db, i.id)]
    if limit is not None:
        items = items[:limit]

    stats = {
        "scanned": len(items),
        "enriched": 0,
        "terminal_miss": 0,
        "transient_deferred": 0,
        "specter2_stored": 0,
        "abstracts_backfilled": 0,
    }
    log.info(
        "enrich-metadata: %d items to enrich (force=%s dry_run=%s workers=%d)",
        len(items), force, dry_run, workers,
    )

    try:
        for batch in _chunks(items, max(1, batch_size)):
            _process_batch(
                db, manifest, batch,
                workers=workers, dry_run=dry_run, stats=stats, log=log,
            )
    finally:
        db.close()

    log.info("enrich-metadata done: %s", stats)
    return stats
