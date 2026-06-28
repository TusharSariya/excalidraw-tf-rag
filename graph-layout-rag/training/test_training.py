"""Unit tests for the SPLADE training data pipeline (run in the training env).

    uv run --no-sync python -m pytest test_training.py -q

Covers (per plan):
  - dedup gate: a known lexical(Jaccard)-leak AND a known semantic(cosine)-leak
    are both dropped, and the assert-0-survivors invariant holds.
  - build_triples: sibling-chunk exclusion + false-negative denoise-drop.
"""
from __future__ import annotations

import numpy as np
import pytest

import build_triples as bt
import gen_train_queries as g


# --------------------------------------------------------------------------- #
# Dedup gate
# --------------------------------------------------------------------------- #
class _FakeEmbedder:
    """Deterministic toy embedder: maps a few known strings to fixed unit vectors
    so we can engineer an exact cosine collision without a real model download.
    Unknown strings get a hashed-but-orthogonal-ish vector.
    """

    DIM = 8

    _TABLE = {
        # The eval query and its paraphrase share a vector -> cosine 1.0 (leak).
        "force directed graph layout with overlap removal": np.array(
            [1, 0, 0, 0, 0, 0, 0, 0], float
        ),
        "pushing nodes apart so they stop colliding in a spring embedding": np.array(
            [1, 0, 0, 0, 0, 0, 0, 0], float
        ),
        # A clearly different training query -> orthogonal (kept).
        "network simplex layer assignment": np.array(
            [0, 1, 0, 0, 0, 0, 0, 0], float
        ),
    }

    def encode(self, texts, **kw):
        out = []
        for t in texts:
            if t in self._TABLE:
                v = self._TABLE[t].copy()
            else:
                h = abs(hash(t)) % (2**32)
                rng = np.random.default_rng(h)
                v = rng.standard_normal(self.DIM)
                v[0] = 0.0  # keep off the "force-directed" axis so unknowns don't leak
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.asarray(out, dtype="float32")


def test_lexical_leak_dropped():
    eval_q = ["barycenter median heuristic crossing reduction two layer"]
    # near-identical wording -> high Jaccard
    leak = {"id": "t1", "query": "barycenter median heuristic for crossing reduction in two layer drawings"}
    safe = {"id": "t2", "query": "scanline constraint compaction in VLSI"}
    survivors, stats = g.dedup_against_eval([leak, safe], eval_q, _FakeEmbedder())
    ids = {r["id"] for r in survivors}
    assert "t1" not in ids, "lexical leak must be dropped"
    assert "t2" in ids
    assert stats["jaccard_dropped"] >= 1


def test_cosine_leak_dropped():
    eval_q = ["force directed graph layout with overlap removal"]
    # lexically disjoint but semantically identical (shares the embedder vector)
    leak = {"id": "c1", "query": "pushing nodes apart so they stop colliding in a spring embedding"}
    safe = {"id": "c2", "query": "network simplex layer assignment"}
    survivors, stats = g.dedup_against_eval([leak, safe], eval_q, _FakeEmbedder())
    ids = {r["id"] for r in survivors}
    assert "c1" not in ids, "semantic (cosine) leak must be dropped"
    assert "c2" in ids
    assert stats["cosine_dropped"] >= 1


def test_assert_zero_survivors_fires_when_filter_bypassed(monkeypatch):
    """If the (patchable) filter gates are loosened so a leak survives filtering,
    the canonical post-filter ASSERT_* safety net must still fire."""
    eval_q = ["force directed graph layout with overlap removal"]
    leak = {"id": "x1", "query": "pushing nodes apart so they stop colliding in a spring embedding"}

    # Loosen ONLY the filter gates (not the canonical assert thresholds), so the
    # cosine-1.0 leak passes filtering and reaches the post-filter assertion.
    monkeypatch.setattr(g, "COSINE_MAX", 1.0001)
    monkeypatch.setattr(g, "JACCARD_MAX", 1.0001)
    # ASSERT_COSINE_MAX / ASSERT_JACCARD_MAX stay at 0.85 / 0.45 -> assert trips.
    with pytest.raises(AssertionError):
        g.dedup_against_eval([leak], eval_q, _FakeEmbedder())


# --------------------------------------------------------------------------- #
# build_triples: sibling exclusion + denoise
# --------------------------------------------------------------------------- #
def test_sibling_exclusion():
    positive_id = "docA:3"
    pos_doc = "docA"
    candidates = [
        ("docA:0", "docA"),   # sibling -> excluded
        ("docA:9", "docA"),   # sibling -> excluded
        ("docB:1", "docB"),   # valid negative
        ("docC:2", "docC"),   # valid negative
        ("docA:3", "docA"),   # the positive itself -> excluded
    ]
    kept = bt.filter_negatives(candidates, positive_id=positive_id, positive_doc=pos_doc)
    kept_ids = [c[0] for c in kept]
    assert "docA:0" not in kept_ids and "docA:9" not in kept_ids
    assert "docA:3" not in kept_ids
    assert kept_ids == ["docB:1", "docC:2"]


def test_denoise_drops_high_scoring_negatives():
    positive_score = 0.90
    scored = [
        ("docB:1", 0.10),   # clearly negative -> keep
        ("docC:2", 0.88),   # within margin of positive -> likely false neg -> DROP
        ("docD:3", 0.50),   # keep
        ("docE:4", 0.89),   # within margin -> DROP
    ]
    kept, dropped = bt.denoise_negatives(scored, positive_score=positive_score, margin=0.05)
    kept_ids = {c[0] for c in kept}
    assert kept_ids == {"docB:1", "docD:3"}
    assert dropped == 2
