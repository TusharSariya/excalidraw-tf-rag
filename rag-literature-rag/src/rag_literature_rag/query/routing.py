"""Query-style routing: keyword vs natural-language vs relational.

Mirrors ``graph_layout_rag.query.routing`` (the two tools keep near-identical query
surfaces). ``classify_query_mode`` is the cheap keyword-vs-NL heuristic; ``is_relational``
/ ``classify_query_regime`` add the P2.2 relational regime that routes clear multi-hop /
relational questions to the graph retrieval arm (``query.graph_arm``). The classifier errs
toward the always-safe default (``keyword``/hybrid); a wrong guess only changes which arm
runs, never correctness.
"""

from __future__ import annotations

# A query is treated as natural-language once it reaches this many whitespace
# tokens, even without an explicit question word.
NL_MIN_TOKENS = 6

# Leading/standalone words that signal a spoken-style question.
QUESTION_WORDS = frozenset(
    {
        "how",
        "what",
        "why",
        "when",
        "where",
        "which",
        "who",
        "whose",
        "can",
        "could",
        "does",
        "do",
        "should",
        "is",
        "are",
        "will",
        "would",
    }
)

Mode = str  # "keyword" | "nl"
Regime = str  # "keyword" | "nl" | "relational"


def classify_query_mode(text: str) -> Mode:
    """Return ``"nl"`` for natural-language queries, ``"keyword"`` otherwise.

    NL if the query ends with ``?``, contains a question word, or has at least
    ``NL_MIN_TOKENS`` tokens. Empty/whitespace input is ``keyword`` (safe local
    default).
    """
    stripped = text.strip()
    if not stripped:
        return "keyword"
    if stripped.endswith("?"):
        return "nl"
    tokens = stripped.split()
    if len(tokens) >= NL_MIN_TOKENS:
        return "nl"
    lowered = {t.strip(".,!?;:").lower() for t in tokens}
    if lowered & QUESTION_WORDS:
        return "nl"
    return "keyword"


# --------------------------------------------------------------------------- #
# Relational regime (P2.2): route clear multi-hop / relational queries to the
# graph retrieval arm. Deliberately conservative — the default stays hybrid, and
# a query only becomes "relational" on an explicit relational cue, never on
# length alone (a long descriptive NL query is still just NL).
# --------------------------------------------------------------------------- #

# Multi-word relational verb/phrase cues (regex, word-boundary anchored). These
# express a *relationship between two things* — lineage, influence, comparison —
# which is exactly what a citation+kNN graph walk answers better than flat hybrid.
_RELATIONAL_PATTERNS = (
    r"\brelate[sd]?\s+to\b",
    r"\brelated\s+to\b",
    r"\brelationship\s+(?:between|of|to|with)\b",
    r"\bconnect(?:s|ed|ion|ions)?\s+(?:to|with|between)\b",
    r"\bbuild[s]?\s+on\b",
    r"\bbuilt\s+on\b",
    r"\bbased\s+on\b",
    r"\bderive[sd]?\s+from\b",
    r"\bdescend[s]?\s+from\b",
    r"\bevolve[sd]?\s+from\b",
    r"\bextend[s]?\b",
    r"\bgenerali[sz]e[sd]?\b",
    r"\binfluence[sd]?\b",
    r"\blineage\b",
    r"\bdifference[s]?\s+between\b",
    r"\bcompare[sd]?\b",
    r"\bcomparison\b",
    r"\b(?:vs\.?|versus)\b",
    r"\bcontrast\b",
    r"\bbridge[sd]?\b",
    r"\bcombine[sd]?\b",
    # "how does X use / leverage / draw on / apply Y"
    r"\bhow\s+(?:do(?:es)?|can|did)\b.*\b(?:use[sd]?|leverage[sd]?|appl(?:y|ies|ied)|draw[s]?\s+on)\b",
    # explicit "between A and B"
    r"\bbetween\b.*\band\b",
)

_RELATIONAL_RE = None  # compiled lazily


def _relational_re():  # noqa: ANN202 — module-private compiled cache
    global _RELATIONAL_RE
    if _RELATIONAL_RE is None:
        import re

        _RELATIONAL_RE = re.compile("|".join(_RELATIONAL_PATTERNS), re.IGNORECASE)
    return _RELATIONAL_RE


def is_relational(text: str) -> bool:
    """True when ``text`` carries a clear relational / multi-hop cue.

    Conservative by design: matches an explicit relational verb/phrase (relate to,
    build on, influence, lineage, compare, difference between, versus, "how does X
    use Y", "between A and B"). Plain topical or single-entity queries — even long
    natural-language ones — return False so they stay on the hybrid arm.
    """
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_relational_re().search(stripped))


def classify_query_regime(text: str) -> Regime:
    """Three-way regime: ``"relational"`` when a relational cue is present, else
    the existing keyword/nl split. Used to decide whether ``query`` should route
    to the graph retrieval arm (relational) or the plain hybrid arm."""
    if is_relational(text):
        return "relational"
    return classify_query_mode(text)
