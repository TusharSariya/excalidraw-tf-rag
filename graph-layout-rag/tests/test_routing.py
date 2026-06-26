import pytest

from graph_layout_rag.query.routing import NL_MIN_TOKENS, classify_query_mode


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
