import pytest
from graph_layout_rag.query.routing import (
    NL_MIN_TOKENS,
    classify_query_mode,
    classify_query_regime,
    is_relational,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        # keyword: short, lexical, no question word
        ("elk", "keyword"),
        ("force directed", "keyword"),
        ("sugiyama layout", "keyword"),
        ("network simplex rank assignment", "keyword"),  # 4 tokens
        ("balanced tree layout edges", "keyword"),  # 4 tokens
        ("one two three four five", "keyword"),  # 5 tokens, below threshold
        ("", "keyword"),  # empty → safe local default
        ("   ", "keyword"),  # whitespace only
        # nl: question word, trailing '?', or >= NL_MIN_TOKENS tokens
        ("how do I keep nodes close", "nl"),  # question word
        ("what is a layered dag drawing", "nl"),  # question word
        ("can edges cross in this drawing", "nl"),  # question word
        ("is this planar?", "nl"),  # question word + '?'
        ("planar?", "nl"),  # trailing '?'
        ("balanced layout for a large directed acyclic graph", "nl"),  # >= 6 tokens
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


def test_question_word_with_punctuation() -> None:
    """Question words are detected after stripping trailing punctuation."""
    assert classify_query_mode("how, exactly") == "nl"


@pytest.mark.parametrize(
    "text,relational",
    [
        # relational / multi-hop -> route to the graph arm
        ("how does stress majorization relate to multidimensional scaling", True),
        ("how does Sugiyama layering build on the network simplex", True),
        ("compare force-directed and spectral graph drawing", True),
        ("relationship between bibliographic coupling and co-citation", True),
        ("how does GraphViz influence modern layout engines", True),
        ("difference between stress majorization and MDS", True),
        ("connection between spectral drawing and stress majorization", True),
        ("edge bundling versus edge routing", True),
        ("is the ELK layout derived from Sugiyama's framework", True),
        ("how does dot use the network simplex for ranking", True),
        ("lineage of orthogonal graph drawing algorithms", True),
        # NOT relational -> stay on the hybrid arm (single-entity / topical)
        ("sugiyama layout", False),
        ("force directed", False),
        ("network simplex rank assignment", False),
        ("what is a layered dag drawing", False),
        ("how do I keep nodes close together", False),
        ("planar?", False),
        ("orthogonal edge routing in large graphs", False),
        ("", False),
    ],
)
def test_is_relational(text: str, relational: bool) -> None:
    assert is_relational(text) is relational


def test_classify_query_regime_three_way() -> None:
    assert classify_query_regime("how does A relate to B") == "relational"
    assert classify_query_regime("what is a layered dag drawing") == "nl"
    assert classify_query_regime("sugiyama layout") == "keyword"


def test_relational_wins_over_nl() -> None:
    # A relational query is also NL-length, but the relational regime takes priority.
    assert classify_query_mode("how does A relate to B for layout") == "nl"
    assert classify_query_regime("how does A relate to B for layout") == "relational"
