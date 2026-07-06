from rag_literature_rag.manifest import (
    Manifest,
    ManifestItem,
    merge_items,
    slug_id,
    titles_similar,
    upsert_item,
)


def test_slug_id():
    assert (
        slug_id("A Technique for Drawing Directed Graphs")
        == "a-technique-for-drawing-directed-graphs"
    )


def test_merge_items_ors_excluded_flag():
    """excluded is sticky across merges, like is_retracted in graph-layout-rag.

    Once any provider observation marks an item excluded=True, a later merge
    from a source that doesn't set it must not silently un-exclude it.
    """
    existing = ManifestItem(
        id="a", title="Paper", source="s1", url="https://example.com/a", status="ok", excluded=True
    )
    incoming = ManifestItem(
        id="a", title="Paper", source="s2", url="https://example.com/a", status="ok", excluded=False
    )
    merged = merge_items(existing, incoming)
    assert merged.excluded is True


def test_upsert_sha256_merge_preserves_losing_id_as_alias():
    """Regression: a merge that re-keys a record must not orphan the old id.

    The Jul-2026 sha256-collapse repair absorbed e.g. arxiv-2004-04906 (DPR)
    into a sibling record and the id vanished entirely — while qrels still
    referenced it, silently corrupting eval metrics. Any merge where the
    incoming id differs from the surviving id must persist the losing id in
    aliasDocIds (which query/identity.py's canonical map resolves).
    """
    manifest = Manifest()
    upsert_item(
        manifest,
        ManifestItem(
            id="arxiv-2004-04906v3",
            title="Dense Passage Retrieval for Open-Domain Question Answering",
            source="arxiv",
            url="https://arxiv.org/pdf/2004.04906v3.pdf",
            status="ok",
            sha256="abc123",
        ),
    )
    upsert_item(
        manifest,
        ManifestItem(
            id="arxiv-2004-04906",
            title="Dense Passage Retrieval for Open-Domain Question Answering",
            source="topic-seed",
            url="https://arxiv.org/pdf/2004.04906.pdf",
            status="ok",
            sha256="abc123",
            externalIds={"ArXiv": "2004.04906"},
        ),
    )
    assert len(manifest.items) == 1
    survivor = manifest.items[0]
    assert survivor.id == "arxiv-2004-04906v3"
    assert "arxiv-2004-04906" in survivor.aliasDocIds
    assert survivor.id not in survivor.aliasDocIds
    # externalIds from the losing record must survive the merge too
    assert survivor.externalIds.get("ArXiv") == "2004.04906"


def test_merge_items_unions_alias_doc_ids():
    existing = ManifestItem(
        id="a",
        title="Paper",
        source="s1",
        url="https://example.com/a",
        status="ok",
        aliasDocIds=["old-1"],
    )
    incoming = ManifestItem(
        id="b",
        title="Paper",
        source="s2",
        url="https://example.com/b",
        status="ok",
        aliasDocIds=["old-2"],
    )
    merged = merge_items(existing, incoming)
    assert set(merged.aliasDocIds) >= {"old-1", "old-2", "a", "b"}


def test_upsert_replaces_by_id():
    manifest = Manifest()
    item = ManifestItem(
        id="test-1",
        title="First",
        source="test",
        url="https://example.com/1",
        status="ok",
    )
    upsert_item(manifest, item)
    assert len(manifest.items) == 1

    updated = item.model_copy(update={"title": "Updated"})
    upsert_item(manifest, updated)
    assert len(manifest.items) == 1
    assert manifest.items[0].title == "Updated"


def test_upsert_merges_provider_provenance_by_doi_without_downgrading_pdf():
    manifest = Manifest()
    upsert_item(
        manifest,
        ManifestItem(
            id="openalex-a",
            title="Paper",
            source="openalex",
            url="https://example.test/paper.pdf",
            localPath="data/raw/pdf/a.pdf",
            status="ok",
            doi="10.1/a",
            tags=["openalex"],
            discoverySources=["openalex"],
            sourceUrls=["https://example.test/paper.pdf"],
        ),
    )
    upsert_item(
        manifest,
        ManifestItem(
            id="crossref-a",
            title="Paper",
            source="crossref",
            url="https://doi.org/10.1/a",
            status="metadata_only",
            doi="https://doi.org/10.1/A",
            abstract="Useful abstract",
            abstractSource="crossref",
            tags=["crossref"],
            discoverySources=["crossref"],
            sourceUrls=["https://doi.org/10.1/a"],
            externalIds={"DOI": "10.1/a"},
        ),
    )

    assert len(manifest.items) == 1
    item = manifest.items[0]
    assert item.id == "openalex-a"
    assert item.status == "ok"
    assert item.localPath == "data/raw/pdf/a.pdf"
    assert item.abstract == "Useful abstract"
    assert item.discoverySources == ["crossref", "openalex"]
    assert item.tags == ["crossref", "openalex"]


# --- title-conflict / DOI-collision regression tests ---
# See data/manifest.json item doi-10-48550-arxiv-2310-11511: a synthesized
# arXiv DOI shared by Self-RAG and an unrelated "CareerX" record produced a
# chimera (Self-RAG's DOI/authors/URL glued to CareerX's title+abstract).


def test_titles_similar_handles_truncation_and_unrelated_titles():
    assert titles_similar(
        "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "Self-RAG: Learning to Retrieve, Generate, and Critique",
    )
    assert not titles_similar(
        "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "CareerX: A Bidirectional Recommendation System for Online Recruitment",
    )
    # Can't compare -> treated as no conflict.
    assert titles_similar(None, "Self-RAG")
    assert titles_similar("Self-RAG", "")


def test_upsert_doi_collision_does_not_clobber_title_or_abstract():
    manifest = Manifest()
    upsert_item(
        manifest,
        ManifestItem(
            id="self-rag",
            title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
            abstract="We introduce Self-Reflective Retrieval-Augmented Generation.",
            authors=["Akari Asai"],
            source="openalex",
            url="https://arxiv.org/abs/2310.11511",
            status="ok",
            doi="10.48550/arxiv.2310.11511",
        ),
    )

    # A higher-status incoming record with a wrongly-synthesized matching DOI
    # but an unrelated title must NOT clobber the existing record.
    upsert_item(
        manifest,
        ManifestItem(
            id="careerx",
            title="CareerX: A Bidirectional Recommendation System for Online Recruitment",
            abstract="CareerX matches candidates to jobs.",
            authors=["Someone Else"],
            source="crossref",
            url="https://doi.org/10.48550/arxiv.2310.11511",
            status="ok",
            doi="10.48550/arxiv.2310.11511",
        ),
    )

    assert len(manifest.items) == 2
    self_rag = next(i for i in manifest.items if i.id == "self-rag")
    assert self_rag.title.startswith("Self-RAG")
    assert self_rag.abstract == "We introduce Self-Reflective Retrieval-Augmented Generation."
    assert self_rag.authors == ["Akari Asai"]
    careerx = next(i for i in manifest.items if i.id == "careerx")
    assert careerx.title.startswith("CareerX")


def test_upsert_doi_match_with_similar_titles_still_merges():
    manifest = Manifest()
    upsert_item(
        manifest,
        ManifestItem(
            id="openalex-a",
            title="Self-RAG: Learning to Retrieve, Generate, and Critique",
            source="openalex",
            url="https://example.test/paper.pdf",
            localPath="data/raw/pdf/a.pdf",
            status="ok",
            doi="10.48550/arxiv.2310.11511",
        ),
    )
    upsert_item(
        manifest,
        ManifestItem(
            id="crossref-a",
            title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
            source="crossref",
            url="https://doi.org/10.48550/arxiv.2310.11511",
            status="metadata_only",
            doi="10.48550/arxiv.2310.11511",
            abstract="Full abstract here",
        ),
    )

    assert len(manifest.items) == 1
    item = manifest.items[0]
    assert item.id == "openalex-a"
    assert item.abstract == "Full abstract here"


def test_merge_items_keeps_existing_title_on_conflict_regardless_of_status():
    existing = ManifestItem(
        id="self-rag",
        title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        abstract="Self-reflective RAG abstract.",
        authors=["Akari Asai"],
        source="openalex",
        url="https://arxiv.org/abs/2310.11511",
        status="metadata_only",
        doi="10.48550/arxiv.2310.11511",
    )
    incoming = ManifestItem(
        id="careerx",
        title="CareerX: A Bidirectional Recommendation System for Online Recruitment",
        abstract="CareerX abstract.",
        authors=["Someone Else"],
        source="crossref",
        url="https://doi.org/10.48550/arxiv.2310.11511",
        status="ok",  # higher status -- would normally win and clobber
        doi="10.48550/arxiv.2310.11511",
    )

    merged = merge_items(existing, incoming)

    assert merged.title == existing.title
    assert merged.abstract == existing.abstract
    assert merged.authors == existing.authors
    # Non-identity fields still merge/backfill normally.
    assert merged.status == "ok"


def test_upsert_sha256_match_with_similar_titles_merges():
    manifest = Manifest()
    upsert_item(
        manifest,
        ManifestItem(
            id="a",
            title="Self-RAG: Learning to Retrieve, Generate",
            source="openalex",
            url="https://example.test/a.pdf",
            status="metadata_only",
            sha256="deadbeef",
        ),
    )
    upsert_item(
        manifest,
        ManifestItem(
            id="b",
            title="Self-RAG: Learning to Retrieve, Generate, and Critique",
            source="crossref",
            url="https://example.test/b.pdf",
            status="ok",
            sha256="deadbeef",
            abstract="filled abstract",
        ),
    )
    assert len(manifest.items) == 1
    assert manifest.items[0].abstract == "filled abstract"


def test_upsert_sha256_match_with_dissimilar_titles_does_not_merge():
    manifest = Manifest()
    upsert_item(
        manifest,
        ManifestItem(
            id="a",
            title="Self-RAG: Learning to Retrieve, Generate, and Critique",
            source="openalex",
            url="https://example.test/a.pdf",
            status="ok",
            sha256="deadbeef",
        ),
    )
    upsert_item(
        manifest,
        ManifestItem(
            id="b",
            title="CareerX: A Bidirectional Recommendation System",
            source="crossref",
            url="https://example.test/b.pdf",
            status="ok",
            sha256="deadbeef",
        ),
    )
    assert len(manifest.items) == 2
