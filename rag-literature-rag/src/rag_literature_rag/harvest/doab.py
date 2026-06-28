"""Directory of Open Access Books (DOAB) full-text harvest (REST search)."""

from __future__ import annotations

import re

import httpx

from rag_literature_rag.harvest.download import download_to_file
from rag_literature_rag.harvest.log import get_logger
from rag_literature_rag.harvest.parallel import parallel_map
from rag_literature_rag.harvest.relevance import is_layout_relevant
from rag_literature_rag.manifest import ManifestItem, relative_local_path, slug_id
from rag_literature_rag.paths import PDF_DIR

DOAB_SEARCH = "https://directory.doabooks.org/rest/search"
DOAB_BASE = "https://directory.doabooks.org"

# Plain-language RAG topics shared by the open-access full-text harvesters.
RAG_TOPIC_QUERIES: list[str] = [
    "retrieval augmented generation",
    "dense passage retrieval",
    "hybrid retrieval reranking",
    "graphrag",
    "agentic rag",
    "rag evaluation",
    "query expansion retrieval",
    "long context rag",
    "chunking retrieval",
    "self-rag corrective rag",
]

_TOPIC_TAGS: dict[str, list[str]] = {
    "retrieval augmented generation": ["foundations"],
    "dense passage retrieval": ["dense-retrieval"],
    "hybrid retrieval reranking": ["hybrid-retrieval", "reranking"],
    "graphrag": ["graphrag"],
    "agentic rag": ["agentic"],
    "rag evaluation": ["evaluation"],
    "query expansion retrieval": ["query-expansion"],
    "long context rag": ["long-context"],
    "chunking retrieval": ["chunking"],
    "self-rag corrective rag": ["self-correcting"],
}

_MAX_OFFSET = 1000


def _metadata_map(item: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for entry in item.get("metadata") or []:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        value = entry.get("value")
        if key and value:
            out.setdefault(key, []).append(str(value).strip())
    return out


def _bitstream_pdf_url(item: dict) -> str | None:
    for bs in item.get("bitstreams") or []:
        if not isinstance(bs, dict):
            continue
        if bs.get("bundleName") != "ORIGINAL" or bs.get("mimeType") != "application/pdf":
            continue
        for md in bs.get("metadata") or []:
            if isinstance(md, dict) and md.get("key") == "oapen.identifier.downloadUrl":
                url = (md.get("value") or "").strip()
                if url:
                    return url
        retrieve = (bs.get("retrieveLink") or "").strip()
        if retrieve:
            return retrieve if retrieve.startswith("http") else f"{DOAB_BASE}{retrieve}"
    return None


def _item_to_spec(item: dict, *, tags: list[str]) -> dict | None:
    title = (item.get("name") or "").strip()
    if not title:
        return None
    md = _metadata_map(item)
    authors = md.get("dc.contributor.author", [])
    year_vals = md.get("dc.date.issued", [])
    year: int | None = None
    if year_vals:
        m = re.search(r"\d{4}", year_vals[0])
        if m:
            year = int(m.group(0))
    abstract = (md.get("dc.description.abstract") or [None])[0]
    identifiers = md.get("dc.identifier", [])
    doi: str | None = None
    for ident in identifiers:
        low = ident.lower()
        if "10." in low and "doi" in low:
            doi = low.split("doi.org/")[-1].strip() if "doi.org/" in low else None
        if low.startswith("10."):
            doi = low
    return {
        "doi": doi,
        "title": re.sub(r"\s+", " ", title),
        "abstract": abstract.strip() if abstract else None,
        "authors": authors,
        "year": year,
        "handle": (item.get("handle") or "").strip() or None,
        "uuid": (item.get("uuid") or "").strip() or None,
        "pdf_url": _bitstream_pdf_url(item),
        "tags": tags,
    }


def _search_doab(topic: str, *, max_results: int, limit: int = 100) -> list[dict]:
    log = get_logger()
    results: list[dict] = []
    offset = 0
    while len(results) < max_results and offset <= _MAX_OFFSET:
        params = {
            "query": topic,
            "expand": "metadata,bitstreams",
            "limit": str(min(limit, max_results - len(results))),
            "offset": str(offset),
        }
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                res = client.get(DOAB_SEARCH, params=params, headers={"Accept": "application/json"})
            if res.status_code != 200:
                log.warning("doab HTTP %s for %r", res.status_code, topic)
                break
            page = res.json()
        except Exception as exc:
            log.warning("doab request failed for %r: %s", topic, exc)
            break
        if not isinstance(page, list) or not page:
            break
        results.extend(page)
        offset += len(page)
        if len(page) < limit:
            break
    return results[:max_results]


def harvest_doab(
    *,
    max_works: int = 200,
    dry_run: bool = False,
    workers: int | None = None,
    existing_ids: set[str] | None = None,
) -> list[ManifestItem]:
    log = get_logger()
    skip_ids = existing_ids or set()
    by_doi: dict[str, dict] = {}
    by_id: dict[str, dict] = {}
    per_topic = max(20, max_works // max(1, len(RAG_TOPIC_QUERIES)))

    try:
        for topic in RAG_TOPIC_QUERIES:
            if len(by_id) >= max_works:
                break
            tags = _TOPIC_TAGS.get(topic, [])
            try:
                items = _search_doab(topic, max_results=per_topic)
            except Exception as exc:
                log.warning("doab search failed for %r: %s", topic, exc)
                continue
            log.info("doab %s: %d candidate books", topic, len(items))
            for raw in items:
                if len(by_id) >= max_works:
                    break
                spec = _item_to_spec(raw, tags=tags)
                if not spec:
                    continue
                if not is_layout_relevant(spec["title"], spec["abstract"]):
                    continue
                key = spec["doi"] or spec["uuid"] or spec["handle"] or slug_id(spec["title"])
                doc_id = slug_id(f"doab-{key}")
                if doc_id in skip_ids or doc_id in by_id:
                    continue
                if spec["doi"] and spec["doi"] in by_doi:
                    continue
                spec["id"] = doc_id
                by_id[doc_id] = spec
                if spec["doi"]:
                    by_doi[spec["doi"]] = spec
    except Exception as exc:
        log.warning("doab harvest discovery failed: %s", exc)

    def _finalize(spec: dict) -> ManifestItem:
        item = ManifestItem(
            id=spec["id"],
            title=spec["title"],
            authors=spec.get("authors", []),
            year=spec.get("year"),
            source="doab",
            url=spec["pdf_url"]
            or (f"https://doi.org/{spec['doi']}" if spec["doi"] else (f"{DOAB_BASE}/handle/{spec['handle']}" if spec["handle"] else "")),
            localPath=None,
            contentType="text/metadata",
            status="metadata_only" if spec["title"] else "failed",
            tags=sorted({*spec["tags"], "doab", "book", "open-access-book"}),
            doi=spec.get("doi"),
            abstract=spec.get("abstract"),
        )
        if dry_run or not spec["pdf_url"]:
            return item
        dest = PDF_DIR / f"{spec['id']}.pdf"
        try:
            dl = download_to_file(
                dest, spec["pdf_url"], doc_id=spec["id"], doi=spec.get("doi"), stage="doab"
            )
            if dl.get("ok") and dest.exists() and dest.read_bytes()[:4] == b"%PDF":
                item.status = "ok"
                item.url = spec["pdf_url"]
                item.sha256 = dl.get("sha256")
                item.contentType = "application/pdf"
                item.localPath = relative_local_path(dest)
                return item
            dest.unlink(missing_ok=True)
        except Exception as exc:
            log.debug("doab download failed for %s: %s", spec["pdf_url"][:120], exc)
            dest.unlink(missing_ok=True)
        return item

    return parallel_map(_finalize, list(by_id.values()), workers=workers, label="doab")
