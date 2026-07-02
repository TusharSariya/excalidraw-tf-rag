"""Offline unit tests for rag_common.citation_matching (pure functions, no network)."""

from __future__ import annotations

from rag_common.citation_matching import (
    best_title_match,
    extract_arxiv_id,
    normalize_title,
    title_similarity,
)

# --- extract_arxiv_id -------------------------------------------------------------


def test_extract_arxiv_id_from_external_ids():
    assert extract_arxiv_id(external_ids={"ArXiv": "2310.11511"}) == "2310.11511"


def test_extract_arxiv_id_from_external_ids_case_insensitive_key():
    assert extract_arxiv_id(external_ids={"arxiv": "2310.11511v2"}) == "2310.11511"


def test_extract_arxiv_id_from_arxiv_style_doi():
    assert extract_arxiv_id(doi="10.48550/arxiv.2310.11511") == "2310.11511"


def test_extract_arxiv_id_from_url_abs():
    assert extract_arxiv_id(url="https://arxiv.org/abs/2310.11511") == "2310.11511"


def test_extract_arxiv_id_from_url_pdf_with_version():
    assert extract_arxiv_id(url="https://arxiv.org/pdf/2310.11511v3") == "2310.11511"


def test_extract_arxiv_id_old_style():
    assert extract_arxiv_id(url="https://arxiv.org/abs/cs.DS/0512345") == "cs.DS/0512345"


def test_extract_arxiv_id_none_when_absent():
    assert extract_arxiv_id(doi="10.1145/foo", url="https://example.com/paper.pdf") is None


def test_extract_arxiv_id_non_arxiv_doi_not_confused():
    assert extract_arxiv_id(doi="10.1109/TVCG.2015.1234") is None


def test_extract_arxiv_id_prefers_external_ids_over_url():
    # externalIds is the most explicit signal; check precedence doesn't crash / picks a
    # consistent id even if both are present and (hypothetically) disagree.
    result = extract_arxiv_id(
        external_ids={"ArXiv": "1111.11111"}, url="https://arxiv.org/abs/2222.22222"
    )
    assert result == "1111.11111"


# --- normalize_title / title_similarity -------------------------------------------


def test_normalize_title_strips_punctuation_and_case():
    assert normalize_title("A Technique for Drawing Directed Graphs!") == (
        "atechniquefordrawingdirectedgraphs"
    )


def test_normalize_title_collapses_whitespace_variants():
    assert normalize_title("Graph   Drawing:  A Survey") == normalize_title(
        "graph-drawing---a-survey"
    )


def test_title_similarity_identical_is_one():
    assert title_similarity("Dense Passage Retrieval", "Dense Passage Retrieval") == 1.0


def test_title_similarity_empty_is_zero():
    assert title_similarity("", "Something") == 0.0
    assert title_similarity("Something", "") == 0.0


def test_title_similarity_robust_to_punctuation_noise():
    a = "ColBERT: Efficient and Effective Passage Search"
    b = "ColBERT Efficient and Effective Passage Search"
    assert title_similarity(a, b) == 1.0


def test_title_similarity_unrelated_titles_low():
    a = "A Technique for Drawing Directed Graphs"
    b = "Dense Passage Retrieval for Open-Domain Question Answering"
    assert title_similarity(a, b) < 0.5


# --- best_title_match ---------------------------------------------------------------


def test_best_title_match_accepts_single_qualifying_candidate():
    candidates = [{"title": "Dense Passage Retrieval", "year": 2020}]
    match = best_title_match(candidates, target_title="Dense Passage Retrieval", target_year=2020)
    assert match is candidates[0]


def test_best_title_match_rejects_when_no_candidate_qualifies():
    candidates = [{"title": "An Unrelated Paper About Compilers", "year": 2020}]
    match = best_title_match(candidates, target_title="Dense Passage Retrieval", target_year=2020)
    assert match is None


def test_best_title_match_rejects_ambiguous_multiple_matches():
    candidates = [
        {"title": "Dense Passage Retrieval", "year": 2020},
        {"title": "Dense Passage Retrieval", "year": 2021},
    ]
    match = best_title_match(
        candidates, target_title="Dense Passage Retrieval", target_year=2020, year_tolerance=5
    )
    assert match is None


def test_best_title_match_year_out_of_tolerance_rejected():
    candidates = [{"title": "Dense Passage Retrieval", "year": 2015}]
    match = best_title_match(
        candidates, target_title="Dense Passage Retrieval", target_year=2020, year_tolerance=1
    )
    assert match is None


def test_best_title_match_year_within_tolerance_accepted():
    candidates = [{"title": "Dense Passage Retrieval", "year": 2021}]
    match = best_title_match(
        candidates, target_title="Dense Passage Retrieval", target_year=2020, year_tolerance=1
    )
    assert match is candidates[0]


def test_best_title_match_missing_year_is_not_penalized():
    candidates = [{"title": "Dense Passage Retrieval", "year": None}]
    match = best_title_match(candidates, target_title="Dense Passage Retrieval", target_year=2020)
    assert match is candidates[0]


def test_best_title_match_custom_keys_for_openalex_shape():
    candidates = [{"display_name": "Dense Passage Retrieval", "publication_year": 2020}]
    match = best_title_match(
        candidates,
        target_title="Dense Passage Retrieval",
        target_year=2020,
        title_key="display_name",
        year_key="publication_year",
    )
    assert match is candidates[0]


def test_best_title_match_empty_target_title_returns_none():
    candidates = [{"title": "Dense Passage Retrieval", "year": 2020}]
    match = best_title_match(candidates, target_title="", target_year=2020)
    assert match is None


def test_best_title_match_empty_candidates_returns_none():
    assert best_title_match([], target_title="Dense Passage Retrieval", target_year=2020) is None
