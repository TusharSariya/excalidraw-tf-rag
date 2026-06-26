"""Regression: `search()` must forward sparse/dense weights to retrieve_candidates.

The natural-language win (+0.07-0.09 nDCG@10) depends on running the 4B index at
a dense-leaning ``sparse_weight≈0.4``. Those weights are accepted by
``retrieve_candidates`` / ``reciprocal_rank_fusion`` but the public ``search()``
wrapper (which ``query_cmd`` calls) historically dropped them — so a query would
silently run at the shipped ``sparse_weight=2.0``. This is the
option-threading-boundary failure class. These tests pin the forwarding so an
accidental edit that drops the weights is caught immediately.
"""

from unittest.mock import patch

from graph_layout_rag.query.hybrid import DENSE_WEIGHT, SPARSE_WEIGHT
from graph_layout_rag.query.search import search


def _spy_retrieve(captured: dict):
    def fake(query, **kwargs):  # noqa: ANN001, ANN202
        captured.update(kwargs)
        return []

    return fake


def test_search_forwards_explicit_weights_to_retrieve() -> None:
    captured: dict = {}
    with patch("graph_layout_rag.query.search.retrieve_candidates", side_effect=_spy_retrieve(captured)):
        search("layered graph", sparse_weight=0.4, dense_weight=1.0)
    assert captured.get("sparse_weight") == 0.4
    assert captured.get("dense_weight") == 1.0


def test_search_defaults_preserve_promoted_weights() -> None:
    """Omitting the kwargs keeps the shipped keyword defaults (sw=2.0/dw=1.0)."""
    captured: dict = {}
    with patch("graph_layout_rag.query.search.retrieve_candidates", side_effect=_spy_retrieve(captured)):
        search("layered graph")
    assert captured.get("sparse_weight") == SPARSE_WEIGHT == 2.0
    assert captured.get("dense_weight") == DENSE_WEIGHT == 1.0
