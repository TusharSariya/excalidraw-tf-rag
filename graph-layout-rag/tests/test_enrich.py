from unittest.mock import patch

from graph_layout_rag.harvest import enrich as enrich_mod
from graph_layout_rag.harvest.enrich import enrich_manifest
from graph_layout_rag.manifest import Manifest, ManifestItem


def _item(**kwargs) -> ManifestItem:
    defaults = {
        "id": "doc",
        "title": "Untitled",
        "source": "dblp",
        "url": "https://example.com",
        "status": "metadata_only",
    }
    defaults.update(kwargs)
    return ManifestItem(**defaults)


def _inverted(text: str) -> dict:
    return {word: [i] for i, word in enumerate(text.split())}


def test_enrich_fills_missing_abstract():
    m = Manifest(
        updatedAt="t",
        items=[
            _item(id="a", doi="10.1/a", abstract=None),
            _item(id="b", doi="10.1/b", abstract="already here"),  # skipped: has abstract
            _item(id="c", doi=None, abstract=None),  # skipped: no doi
        ],
    )

    with (
        patch.object(enrich_mod, "_fetch_abstract", return_value="filled in abstract"),
        patch.object(enrich_mod, "_fetch_title", return_value=None),
    ):
        stats = enrich_manifest(m, workers=1)

    assert stats["candidates"] == 1  # only item a
    assert stats["enriched"] == 1
    by_id = {i.id: i for i in m.items}
    assert by_id["a"].abstract == "filled in abstract"
    assert by_id["a"].abstractSource == "enrich"
    assert by_id["b"].abstract == "already here"  # untouched
    assert by_id["c"].abstract is None


def test_enrich_dry_run_does_not_mutate():
    m = Manifest(updatedAt="t", items=[_item(id="a", doi="10.1/a", abstract=None)])

    with (
        patch.object(enrich_mod, "_fetch_abstract", return_value="node overlap removal"),
        patch.object(enrich_mod, "_fetch_title", return_value=None),
    ):
        stats = enrich_manifest(m, workers=1, dry_run=True)

    assert stats["enriched"] == 1
    assert m.items[0].abstract is None  # not written in dry-run


def test_enrich_handles_no_provider_hit():
    m = Manifest(updatedAt="t", items=[_item(id="a", doi="10.1/a", abstract=None)])

    with (
        patch.object(enrich_mod, "_fetch_abstract", return_value=None),
        patch.object(enrich_mod, "_fetch_title", return_value=None),
    ):
        stats = enrich_manifest(m, workers=1)

    assert stats["candidates"] == 1
    assert stats["enriched"] == 0
    assert m.items[0].abstract is None


def test_enrich_skips_abstract_on_title_mismatch():
    # DOI lookup resolves to a different paper's record (e.g. a synthesized
    # arXiv DOI collision) -- must not attach that record's abstract.
    m = Manifest(
        updatedAt="t",
        items=[
            _item(
                id="self-rag",
                title="Self-RAG: Learning to Retrieve, Generate, and Critique",
                doi="10.48550/arxiv.2310.11511",
                abstract=None,
            ),
        ],
    )

    with (
        patch.object(enrich_mod, "_fetch_abstract", return_value="CareerX matches candidates."),
        patch.object(
            enrich_mod,
            "_fetch_title",
            return_value="CareerX: A Bidirectional Recommendation System",
        ),
    ):
        stats = enrich_manifest(m, workers=1)

    assert stats["candidates"] == 1
    assert stats["enriched"] == 0
    assert m.items[0].abstract is None
    assert m.items[0].abstractSource is None


def test_enrich_fills_abstract_when_fetched_title_matches():
    m = Manifest(
        updatedAt="t",
        items=[
            _item(
                id="self-rag",
                title="Self-RAG: Learning to Retrieve, Generate, and Critique",
                doi="10.48550/arxiv.2310.11511",
                abstract=None,
            ),
        ],
    )

    with (
        patch.object(enrich_mod, "_fetch_abstract", return_value="Self-reflective RAG abstract."),
        patch.object(
            enrich_mod,
            "_fetch_title",
            return_value="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        ),
    ):
        stats = enrich_manifest(m, workers=1)

    assert stats["enriched"] == 1
    assert m.items[0].abstract == "Self-reflective RAG abstract."
    assert m.items[0].abstractSource == "enrich"


def test_enrich_no_candidates():
    m = Manifest(updatedAt="t", items=[_item(id="a", doi=None, abstract=None)])
    stats = enrich_manifest(m, workers=1)
    assert stats == {"scanned": 1, "candidates": 0, "enriched": 0, "skipped": 0}


# --- provider chain ---


def test_openalex_decodes_inverted_index():
    with patch.object(
        enrich_mod,
        "_fetch_by_doi",
        return_value={"abstract_inverted_index": _inverted("compaction constraint graph")},
    ):
        assert enrich_mod._from_openalex("10.1/a") == "compaction constraint graph"


def test_fetch_abstract_falls_through_to_crossref_then_s2():
    # OpenAlex misses, Crossref misses, S2 hits — chain returns S2's abstract.
    with (
        patch.object(enrich_mod, "_from_openalex", return_value=None),
        patch.object(enrich_mod, "_from_crossref", return_value=None),
        patch.object(enrich_mod, "_from_semantic_scholar", return_value="s2 abstract"),
    ):
        assert enrich_mod._fetch_abstract("10.1/a") == "s2 abstract"


def test_fetch_abstract_stops_at_first_hit():
    # OpenAlex hits — Crossref/S2 must not be consulted.
    with (
        patch.object(enrich_mod, "_from_openalex", return_value="oa abstract"),
        patch.object(enrich_mod, "_from_crossref") as cr,
        patch.object(enrich_mod, "_from_semantic_scholar") as s2,
    ):
        assert enrich_mod._fetch_abstract("10.1/a") == "oa abstract"
        cr.assert_not_called()
        s2.assert_not_called()
