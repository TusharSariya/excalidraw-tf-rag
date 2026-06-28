"""Unit tests for late-chunking pooling math (no model load)."""
from __future__ import annotations

import math

from rag_common.local_embed import pool_span


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def test_pool_span_mean_selects_intersecting_real_tokens():
    # tokens 0,1 are real and inside [0,4); token 2 is special (0,0); token 3
    # starts at 4 so does not intersect [0,4).
    vecs = [[1.0, 0.0], [3.0, 0.0], [0.0, 0.0], [5.0, 5.0]]
    offs = [(0, 2), (2, 4), (0, 0), (4, 8)]
    out = pool_span(vecs, offs, 0, 4)
    # mean([1,0],[3,0]) = [2,0] -> normalized [1,0]
    assert out == [1.0, 0.0]
    assert abs(_norm(out) - 1.0) < 1e-6


def test_pool_span_truncates_then_normalizes():
    vecs = [[3.0, 4.0, 9.0, 9.0], [3.0, 4.0, 9.0, 9.0]]
    offs = [(0, 1), (1, 2)]
    out = pool_span(vecs, offs, 0, 2, target_dim=2)
    assert len(out) == 2
    # mean is [3,4,9,9]; truncate->[3,4]; /5 -> [0.6,0.8]
    assert abs(out[0] - 0.6) < 1e-6
    assert abs(out[1] - 0.8) < 1e-6
    assert abs(_norm(out) - 1.0) < 1e-6


def test_pool_span_empty_span_falls_back_to_real_tokens():
    vecs = [[1.0, 0.0], [3.0, 0.0], [0.0, 0.0]]
    offs = [(0, 2), (2, 4), (0, 0)]
    out = pool_span(vecs, offs, 100, 200)  # selects nothing -> fallback real
    # mean([1,0],[3,0]) -> [2,0] -> [1,0]
    assert out == [1.0, 0.0]


def test_pool_span_skips_special_tokens():
    vecs = [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]
    offs = [(0, 0), (0, 3), (0, 0)]
    out = pool_span(vecs, offs, 0, 3)
    assert out == [1.0, 0.0]


def test_pool_span_empty_input_returns_empty():
    assert pool_span([], [], 0, 5) == []


def test_pool_span_max_method():
    vecs = [[1.0, 0.0], [0.0, 4.0]]
    offs = [(0, 2), (2, 4)]
    out = pool_span(vecs, offs, 0, 4, method="max")
    # max -> [1,4] -> normalized
    n = math.sqrt(1 + 16)
    assert abs(out[0] - 1.0 / n) < 1e-6
    assert abs(out[1] - 4.0 / n) < 1e-6
