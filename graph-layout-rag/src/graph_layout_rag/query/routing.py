"""Query-style routing: keyword vs natural-language.

The graph-layout corpus is served by two retrieval regimes (see
``docs/qwen-ladder-campaign-audit-log.md`` M10-M16):

- **keyword / LLM-issued** queries (high lexical overlap) → local 0.6B index,
  sparse-heavy hybrid (``sparse_weight≈2.0``).
- **human / natural-language** queries (low overlap) → 4B index, dense-leaning
  hybrid (``sparse_weight≈0.4``); +0.07-0.09 nDCG@10 over 0.6B.

``classify_query_mode`` is a cheap heuristic that picks the regime so a single
entry point (``scripts/query_auto.sh``) can dispatch to the right backend. It
errs toward ``keyword`` (the always-available local path) on short inputs; a
wrong guess only changes which backend runs, never correctness, and the
dispatcher prints the chosen backend so the caller can override with ``--mode``.
"""

from __future__ import annotations

# A query is treated as natural-language once it reaches this many whitespace
# tokens, even without an explicit question word ("balanced layout for a large
# directed acyclic graph" reads as NL).
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
