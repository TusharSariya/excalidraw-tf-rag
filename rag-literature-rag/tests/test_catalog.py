from rag_literature_rag.catalog.classify import build_catalog, classify_item
from rag_literature_rag.catalog.taxonomy import categories_from_tags
from rag_literature_rag.manifest import Manifest, ManifestItem


def _item(**kwargs) -> ManifestItem:
    defaults = {
        "id": "test-doc",
        "title": "Untitled",
        "source": "test",
        "url": "https://example.com",
        "status": "ok",
    }
    defaults.update(kwargs)
    return ManifestItem(**defaults)


def test_self_rag_by_tag():
    item = _item(
        id="self-rag",
        title="Self-RAG: Learning to Retrieve, Generate, and Critique",
        tags=["self-correcting", "reflection"],
    )
    categories, methods = classify_item(item)
    assert "self-correcting" in categories
    assert methods[categories.index("self-correcting")] == "tag"


def test_graphrag_by_tag():
    item = _item(
        title="From Local to Global: A Graph RAG Approach",
        tags=["graphrag", "community"],
    )
    categories, _ = classify_item(item)
    assert "graphrag" in categories


def test_keyword_fallback_hybrid_retrieval():
    item = _item(
        title="Hybrid dense sparse retrieval with reciprocal rank fusion",
        tags=["openalex"],
        abstract="We combine BM25 and dense embeddings using RRF.",
    )
    categories, methods = classify_item(item)
    assert "hybrid-retrieval" in categories
    assert methods[categories.index("hybrid-retrieval")] == "keyword"


def test_multi_category_agentic_and_evaluation():
    item = _item(
        title="SoK: Agentic RAG evaluation trajectory formalization",
        tags=["agentic", "sok", "evaluation"],
    )
    categories, _ = classify_item(item)
    assert "agentic" in categories
    assert "evaluation" in categories or "survey" in categories


def test_build_catalog_skips_excluded_items(monkeypatch, tmp_path):
    """excluded=True items must never surface in the query-facing catalog.

    build_catalog() is what retrieve.py's catalog_maps() reads live from
    data/manifest.json -- marking an item excluded (the P1c biomedical
    re-score exclusion mechanism) must take effect without re-ingesting.
    """
    kept = _item(id="kept-doc", title="Retrieval-Augmented Generation Survey")
    dropped = _item(
        id="dropped-doc",
        title="Working memory in episodic long-term memory",
        excluded=True,
    )
    manifest = Manifest(items=[kept, dropped])
    monkeypatch.setattr("rag_literature_rag.catalog.classify.load_manifest", lambda: manifest)
    monkeypatch.setattr("rag_literature_rag.catalog.classify._has_pdf_on_disk", lambda item: True)
    entries = build_catalog(status="ok")
    ids = {e.doc_id for e in entries}
    assert "kept-doc" in ids
    assert "dropped-doc" not in ids


def test_uncategorized_generic_title():
    item = _item(
        title="R: A Language and Environment for Statistical Computing",
        tags=["openalex", "rag-literature"],
    )
    categories, _ = classify_item(item)
    assert categories == []


def test_tag_maps_for_rag_pipeline():
    assert "dense-retrieval" in categories_from_tags(["late-interaction"])
    assert "query-expansion" in categories_from_tags(["query-rewriting"])
    assert "chunking" in categories_from_tags(["hierarchical-index"])
    cats = categories_from_tags(["graphrag", "community"])
    assert "graphrag" in cats
