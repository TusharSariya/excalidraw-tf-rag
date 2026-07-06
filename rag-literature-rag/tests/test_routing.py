import pytest
from rag_literature_rag.query.routing import (
    NL_MIN_TOKENS,
    classify_query_mode,
    classify_query_regime,
    is_relational,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        # keyword: short, lexical, no question word
        ("colbert", "keyword"),
        ("dense passage retrieval", "keyword"),
        ("late interaction retrieval", "keyword"),
        ("bm25 sparse baseline", "keyword"),  # 3 tokens
        ("one two three four five", "keyword"),  # 5 tokens, below threshold
        ("", "keyword"),  # empty -> safe local default
        ("   ", "keyword"),  # whitespace only
        # nl: question word, trailing '?', or >= NL_MIN_TOKENS tokens
        ("how do I rerank retrieved passages", "nl"),  # question word
        ("what is a bi-encoder for retrieval", "nl"),  # question word
        ("is this a cross encoder?", "nl"),  # question word + '?'
        ("retrievable?", "nl"),  # trailing '?'
        ("one two three four five six", "nl"),  # exactly threshold tokens
    ],
)
def test_classify_query_mode(text: str, expected: str) -> None:
    assert classify_query_mode(text) == expected


def test_token_threshold_boundary() -> None:
    """A query crosses to NL exactly at NL_MIN_TOKENS tokens (no question word)."""
    below = " ".join(["lex"] * (NL_MIN_TOKENS - 1))
    at = " ".join(["lex"] * NL_MIN_TOKENS)
    assert classify_query_mode(below) == "keyword"
    assert classify_query_mode(at) == "nl"


@pytest.mark.parametrize(
    "text,relational",
    [
        # relational / multi-hop -> route to the graph arm
        ("how does ColBERT build on BERT for retrieval", True),
        ("how does GraphRAG relate to personalized pagerank", True),
        ("compare DPR and BM25 for open-domain QA", True),
        ("relationship between HyDE and query expansion", True),
        ("how did DPR influence later dense retrieval", True),
        ("difference between bi-encoders and cross-encoders", True),
        ("connection between RAG and retrieval augmented generation", True),
        ("SPLADE versus ColBERT", True),
        ("agentic RAG derived from ReAct prompting", True),
        ("how does HippoRAG use a knowledge graph", True),
        ("lineage of dense retrieval models", True),
        # NOT relational -> stay on the hybrid arm (single-entity / topical)
        ("colbert", False),
        ("dense passage retrieval", False),
        ("late interaction retrieval", False),
        ("what is a bi-encoder for retrieval", False),
        ("how do I rerank retrieved passages", False),
        ("retrievable?", False),
        ("contrastive training for retrievers", False),
        ("", False),
    ],
)
def test_is_relational(text: str, relational: bool) -> None:
    assert is_relational(text) is relational


def test_classify_query_regime_three_way() -> None:
    assert classify_query_regime("how does ColBERT relate to BERT") == "relational"
    assert classify_query_regime("what is a bi-encoder for retrieval") == "nl"
    assert classify_query_regime("dense passage retrieval") == "keyword"


def test_relational_wins_over_nl() -> None:
    # A relational query is also NL-length, but the relational regime takes priority.
    assert classify_query_mode("how does ColBERT build on BERT for retrieval") == "nl"
    assert classify_query_regime("how does ColBERT build on BERT for retrieval") == "relational"
