"""Semantic Scholar open-access PDF harvest (bulk search endpoint).

Named s2orc per the harvest plan, but this uses the per-paper ``openAccessPdf.url``
from the Semantic Scholar Graph bulk search rather than the bulk S2ORC full-text
dataset (which is download-only).
"""

from __future__ import annotations

import os
import re

from rag_literature_rag.harvest.download import download_to_file
from rag_literature_rag.harvest.log import get_logger
from rag_literature_rag.harvest.parallel import parallel_map
from rag_literature_rag.harvest.providers import SEMANTIC_SCHOLAR, OutcomeKind
from rag_literature_rag.harvest.relevance import is_layout_relevant
from rag_literature_rag.manifest import ManifestItem, relative_local_path, slug_id
from rag_literature_rag.paths import PDF_DIR

S2_BULK_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "title,year,abstract,isOpenAccess,openAccessPdf,externalIds,authors"

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


def _s2_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = (os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY") or "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _authors(paper: dict) -> list[str]:
    out: list[str] = []
    for a in paper.get("authors") or []:
        name = (a.get("name") if isinstance(a, dict) else None) or ""
        name = name.strip()
        if name:
            out.append(name)
    return out


def _pdf_url(paper: dict) -> str | None:
    oa = paper.get("openAccessPdf") or {}
    url = oa.get("url") if isinstance(oa, dict) else None
    return url.strip() if isinstance(url, str) and url.strip() else None


def _paper_to_spec(paper: dict, *, tags: list[str]) -> dict | None:
    title = (paper.get("title") or "").strip()
    if not title:
        return None
    external = paper.get("externalIds") or {}
    doi = (external.get("DOI") or "").strip().lower() or None
    year = paper.get("year")
    return {
        "doi": doi,
        "title": re.sub(r"\s+", " ", title),
        "abstract": (paper.get("abstract") or "").strip() or None,
        "authors": _authors(paper),
        "year": year if isinstance(year, int) else None,
        "pdf_url": _pdf_url(paper),
        "tags": tags,
    }


def _search_s2(topic: str, *, max_results: int) -> list[dict]:
    log = get_logger()
    results: list[dict] = []
    headers = _s2_headers()
    token: str | None = None
    while len(results) < max_results:
        params = {
            "query": topic,
            "fields": S2_FIELDS,
            "openAccessPdf": "",
            "limit": "1000",
        }
        if token:
            params["token"] = token
        outcome = SEMANTIC_SCHOLAR.request(
            "GET", S2_BULK_SEARCH, params=params, headers=headers, timeout=60.0
        )
        if outcome.kind is not OutcomeKind.SUCCESS:
            log.warning("s2 search %r stopped: %s", topic, outcome.kind.value)
            break
        data = outcome.data or {}
        page = data.get("data") or []
        if not page:
            break
        results.extend(page)
        token = data.get("token")
        if not token:
            break
    return results[:max_results]


def harvest_s2orc(
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
                papers = _search_s2(topic, max_results=per_topic)
            except Exception as exc:
                log.warning("s2 search failed for %r: %s", topic, exc)
                continue
            log.info("s2orc %s: %d candidate papers", topic, len(papers))
            for paper in papers:
                if len(by_id) >= max_works:
                    break
                spec = _paper_to_spec(paper, tags=tags)
                if not spec:
                    continue
                if not is_layout_relevant(spec["title"], spec["abstract"]):
                    continue
                key = spec["doi"] or slug_id(spec["title"])
                doc_id = slug_id(f"s2orc-{key}")
                if doc_id in skip_ids or doc_id in by_id:
                    continue
                if spec["doi"] and spec["doi"] in by_doi:
                    continue
                spec["id"] = doc_id
                by_id[doc_id] = spec
                if spec["doi"]:
                    by_doi[spec["doi"]] = spec
    except Exception as exc:
        log.warning("s2orc harvest discovery failed: %s", exc)

    def _finalize(spec: dict) -> ManifestItem:
        item = ManifestItem(
            id=spec["id"],
            title=spec["title"],
            authors=spec.get("authors", []),
            year=spec.get("year"),
            source="s2orc",
            url=spec["pdf_url"] or (f"https://doi.org/{spec['doi']}" if spec["doi"] else ""),
            localPath=None,
            contentType="text/metadata",
            status="metadata_only" if (spec["title"] and spec["doi"]) else "failed",
            tags=sorted({*spec["tags"], "s2orc", "semantic-scholar", "open-access"}),
            doi=spec.get("doi"),
            abstract=spec.get("abstract"),
        )
        if dry_run or not spec["pdf_url"]:
            return item
        dest = PDF_DIR / f"{spec['id']}.pdf"
        try:
            dl = download_to_file(
                dest, spec["pdf_url"], doc_id=spec["id"], doi=spec.get("doi"), stage="s2orc"
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
            log.debug("s2orc download failed for %s: %s", spec["pdf_url"][:120], exc)
            dest.unlink(missing_ok=True)
        return item

    return parallel_map(_finalize, list(by_id.values()), workers=workers, label="s2orc")
