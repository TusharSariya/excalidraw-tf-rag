"""Probe which SPLADE bases actually load as a SparseEncoder in this env, and
whether they need trust_remote_code (which the eval harness does NOT pass)."""
from __future__ import annotations

import json
import sys

import torch
from sentence_transformers import SparseEncoder

DEV = "cuda" if torch.cuda.is_available() else "cpu"

CANDIDATES = [
    ("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte", False),
    ("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte", True),
    ("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill", False),
    ("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill", False),
    ("naver/splade-v3", False),
    ("naver/splade-v3-distilbert", False),
    ("prithivida/Splade_PP_en_v1", False),
]


def main() -> int:
    results = []
    for name, trc in CANDIDATES:
        rec = {"model": name, "trust_remote_code": trc}
        try:
            kw = {"device": DEV}
            if trc:
                kw["trust_remote_code"] = True
            m = SparseEncoder(name, **kw)
            emb = m.encode_document(["network simplex layered graph drawing"],
                                    convert_to_sparse_tensor=True)
            rec["nnz"] = int(emb[0].coalesce().values().numel())
            rec["ok"] = True
            del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as exc:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        results.append(rec)
        print(json.dumps(rec), flush=True)
    print("PROBE_RESULTS:", json.dumps(results), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
