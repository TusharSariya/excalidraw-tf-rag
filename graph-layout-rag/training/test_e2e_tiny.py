"""[->E2E] Tiny-corpus smoke test (plan-required).

~20 synthetic chunks -> build contrastive triples -> 1-step LoRA train -> merge ->
load merged via SparseEncoder -> assert non-empty sparse encoding + served != base.

Runs end-to-end on the desktop GPU env:
    uv run --no-sync python -m pytest test_e2e_tiny.py -q -s

Marked slow: downloads/loads the SPLADE base + runs one optimizer step. Skips
gracefully if the training stack isn't importable (so the fast unit suite stays green
on machines without it).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

st = pytest.importorskip("sentence_transformers")
pytest.importorskip("peft")

import train_splade as T  # noqa: E402

BASE = T.BASE_MODEL

# 20 tiny graph-drawing "chunks": (chunk_id, doc_id, text).
CHUNKS = [
    ("d0:0", "d0", "Network simplex assigns ranks to nodes in a layered digraph minimizing total edge length."),
    ("d1:0", "d1", "The barycenter heuristic reduces edge crossings between two adjacent layers of a Sugiyama drawing."),
    ("d2:0", "d2", "Force-directed layout places nodes by simulating springs on edges and repulsion between nodes."),
    ("d3:0", "d3", "Stress majorization minimizes a stress energy to embed a graph respecting target distances."),
    ("d4:0", "d4", "Orthogonal edge routing bends arrows along axis-aligned segments to avoid node overlaps."),
    ("d5:0", "d5", "Compound graph layout nests clusters and lays out child subgraphs inside parent borders."),
    ("d6:0", "d6", "VPSC solves separation constraints to remove node overlaps while preserving layout structure."),
    ("d7:0", "d7", "Coffman-Graham layering bounds the width of a layered drawing given a maximum per-layer node count."),
    ("d8:0", "d8", "Sifting iteratively moves a vertex to its locally optimal position to reduce crossings."),
    ("d9:0", "d9", "Longest-path layering assigns each node the rank equal to its longest path from a source."),
    ("d10:0", "d10", "Dummy nodes are inserted on long edges so every edge spans exactly one layer in Sugiyama."),
    ("d11:0", "d11", "ELK provides a layered algorithm with port constraints for hierarchical diagram layout."),
    ("d12:0", "d12", "Reciprocal rank fusion combines rankings from BM25 and dense retrieval into one ordering."),
    ("d13:0", "d13", "Scanline constraint compaction packs a VLSI layout by sweeping a line over a constraint graph."),
    ("d14:0", "d14", "Edge bundling groups nearly-parallel edges to reduce visual clutter in dense graphs."),
    ("d15:0", "d15", "Treemaps recursively subdivide a rectangle to show hierarchical data by area."),
    ("d16:0", "d16", "Spectral layout positions nodes using eigenvectors of the graph Laplacian matrix."),
    ("d17:0", "d17", "Crossing minimization between two layers is NP-hard even for a fixed order on one side."),
    ("d18:0", "d18", "Node promotion reduces dummy nodes by raising a node above its current layer assignment."),
    ("d19:0", "d19", "Overlap removal nudges overlapping rectangles apart along the x then y axes."),
]


def _make_triples(path: Path) -> int:
    """Extractive triples: query = first clause of each chunk; positive = its text;
    negatives = a few other chunks (different docs). No LLM, no GPU."""
    rows = []
    for i, (cid, doc, text) in enumerate(CHUNKS):
        query = text.split(",")[0][:60]  # short extractive anchor
        negs = [CHUNKS[(i + k) % len(CHUNKS)] for k in (3, 5, 7)]
        negs = [n for n in negs if n[1] != doc][:3]
        rows.append({
            "id": f"train-v1:{cid}",
            "query": query,
            "positive_text": text,
            "negative_texts": [n[2] for n in negs],
            "positive_chunk_id": cid,
            "negative_chunk_ids": [n[0] for n in negs],
        })
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


@pytest.mark.slow
def test_tiny_e2e_train_merge_serve():
    import torch
    from sentence_transformers import SparseEncoder

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        triples = td / "triples.jsonl"
        n = _make_triples(triples)
        assert n >= 18

        out = td / "ckpt"

        # 1-step train via the real entrypoint (argv-driven).
        import sys
        argv = [
            "train_splade.py",
            "--triples", str(triples),
            "--out", str(out),
            "--max-steps", "1",
            "--batch-size", "8",
            "--eval-frac", "0",
            "--warmup-ratio", "0",
        ]
        old = sys.argv
        sys.argv = argv
        try:
            rc = T.main()
        finally:
            sys.argv = old
        assert rc == 0

        summary = json.loads((out / "train_summary.json").read_text())
        assert summary["served_not_base"] is True
        assert summary["served_vs_base_max_abs_delta"] > 1e-6
        assert summary["merged_nnz"] > 0

        # Independent reload + non-empty sparse encoding.
        m = SparseEncoder(str(out), device="cuda" if torch.cuda.is_available() else "cpu")
        emb = m.encode_document(["how do I assign layers to nodes in a digraph"],
                                convert_to_sparse_tensor=True)
        assert int(emb[0].coalesce().values().numel()) > 0
