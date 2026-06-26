from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import click

from graph_layout_rag.eval.encode_device import EncodeDevice, fastembed_kwargs, resolve_encode_device
from graph_layout_rag.eval.hardware import assert_memory_start_safe, memory_snapshot
from graph_layout_rag.eval.splade_v3_encoder import SpladeV3Encoder, is_pytorch_sparse_model
from graph_layout_rag.paths import CHUNKS_TABLE, RETRIEVAL_INDEXES_DIR, profile_index_paths

IndexKind = Literal["splade", "colbert"]
INDEX_KINDS: tuple[IndexKind, ...] = ("splade", "colbert")
DEFAULT_MODELS = {
    "splade": "prithivida/Splade_PP_en_v1",
    "colbert": "answerdotai/answerai-colbert-small-v1",
}


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-")


def _corpus_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("id") or "").encode())
        digest.update(str(row.get("canonical_sha256") or "").encode())
        digest.update((row.get("text") or "").encode())
    return digest.hexdigest()[:16]


def _load_rows(profile: str) -> list[dict[str, Any]]:
    import lancedb

    paths = profile_index_paths(profile)
    db = lancedb.connect(str(paths.lance_dir))
    rows = db.open_table(CHUNKS_TABLE).to_arrow().to_pylist()
    return [{k: v for k, v in row.items() if k != "vector"} for row in rows]


def _clean_payload(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and value != value:
            value = None
        out[key] = value
    return out


def _qdrant():
    try:
        from qdrant_client import QdrantClient, models
        from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
    except ImportError as exc:
        raise click.ClickException(
            "Retrieval experiment dependencies are missing. "
            "Run: uv sync --extra retrieval-experiments  "
            "(GPU host: uv sync --extra retrieval-experiments-gpu)"
        ) from exc
    return QdrantClient, models, SparseTextEmbedding, LateInteractionTextEmbedding


def _sparse_backend(model_name: str) -> Literal["fastembed", "pytorch_sparse"]:
    return "pytorch_sparse" if is_pytorch_sparse_model(model_name) else "fastembed"


def _sparse_encoder(model_name: str, *, encode_device: EncodeDevice | None):
    if _sparse_backend(model_name) == "pytorch_sparse":
        device = resolve_encode_device(encode_device)
        return SpladeV3Encoder(model_name, device=device)
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=model_name, **fastembed_kwargs(encode_device))


def _fastembed_supports_colbert(model_name: str) -> bool:
    from fastembed import LateInteractionTextEmbedding

    return any(
        m.get("model") == model_name
        for m in LateInteractionTextEmbedding.list_supported_models()
    )


class _PylateColbertEncoder:
    """Late-interaction encoder for ColBERT checkpoints not shipped in fastembed
    (e.g. mxbai-edge-colbert). Exposes the same ``embed``/``query_embed`` surface
    the build/search paths use, yielding per-text 2D token-embedding arrays."""

    def __init__(self, model_name: str, *, encode_device: EncodeDevice | None) -> None:
        try:
            from pylate import models as pylate_models
        except ImportError as exc:
            raise click.ClickException(
                f"ColBERT model {model_name!r} is not shipped in fastembed and requires pylate. "
                "Run: uv sync --extra retrieval-experiments-gpu"
            ) from exc
        import torch

        device = resolve_encode_device(encode_device)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.model_name = model_name
        self._model = pylate_models.ColBERT(model_name_or_path=model_name, device=device)

    def _encode(self, texts: list[str], *, batch_size: int, is_query: bool):
        import numpy as np

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            is_query=is_query,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for emb in embeddings:
            yield np.asarray(emb, dtype=np.float32)

    def embed(self, texts: list[str], *, batch_size: int):
        yield from self._encode(list(texts), batch_size=batch_size, is_query=False)

    def query_embed(self, query: str):
        yield from self._encode([query], batch_size=1, is_query=True)


def _colbert_encoder(model_name: str, *, encode_device: EncodeDevice | None):
    if _fastembed_supports_colbert(model_name):
        from fastembed import LateInteractionTextEmbedding

        return LateInteractionTextEmbedding(model_name=model_name, **fastembed_kwargs(encode_device))
    return _PylateColbertEncoder(model_name, encode_device=encode_device)


def _connect(QdrantClient, index_dir: Path, index_id: str, *, meta: dict | None = None):
    """Connect to Qdrant. Server (out-of-process, no OOM) when GRAPH_RAG_QDRANT_URL
    is set — uses a per-index collection so multiple indexes coexist; else local
    on-disk mode (one 'chunks' collection per index dir).

    Returns (client, collection_name).
    """
    if meta is not None:
        url = meta.get("qdrant_url") or os.getenv("GRAPH_RAG_QDRANT_URL")
        collection = meta.get("collection", "chunks")
    else:
        url = os.getenv("GRAPH_RAG_QDRANT_URL")
        collection = index_id if url else "chunks"
    if url:
        return QdrantClient(url=url, timeout=120), collection
    return QdrantClient(path=str(index_dir / "qdrant")), collection


def build_experimental_index(
    *,
    base_profile: str,
    kind: IndexKind,
    model_name: str | None = None,
    batch_size: int = 4,
    min_available_gb: float = 8.0,
    encode_device: EncodeDevice | None = None,
) -> Path:
    if kind not in INDEX_KINDS:
        raise ValueError(f"Unknown index kind {kind!r}")
    start_memory = assert_memory_start_safe(min_available_gb)
    rows = _load_rows(base_profile)
    model_name = model_name or DEFAULT_MODELS[kind]
    if kind == "colbert" and is_pytorch_sparse_model(model_name):
        raise ValueError(f"Model {model_name!r} is a PyTorch sparse model; use --kind splade")
    fingerprint = _corpus_fingerprint(rows)
    index_id = f"{kind}-{_slug(model_name)}-{fingerprint}-{_now_id()}"
    index_dir = RETRIEVAL_INDEXES_DIR / _slug(base_profile) / index_id
    if index_dir.exists():
        raise FileExistsError(f"Refusing to overwrite experimental index: {index_dir}")
    index_dir.mkdir(parents=True)

    QdrantClient, models, _SparseTextEmbedding, _LateInteractionTextEmbedding = _qdrant()
    client, collection = _connect(QdrantClient, index_dir, index_id)
    started = time.monotonic()
    resolved_device = resolve_encode_device(encode_device)

    texts = [(row.get("text") or "")[:8000] for row in rows]
    sparse_backend = _sparse_backend(model_name) if kind == "splade" else None
    if kind == "splade":
        model = _sparse_encoder(model_name, encode_device=encode_device)
        client.create_collection(
            collection_name=collection,
            vectors_config={},
            sparse_vectors_config={
                "splade": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                )
            },
        )
        embedded = model.embed(texts, batch_size=batch_size)
        points = (
            models.PointStruct(
                id=idx,
                payload=_clean_payload(row),
                vector={
                    "splade": models.SparseVector(
                        indices=vector.indices.tolist(),
                        values=vector.values.tolist(),
                    )
                },
            )
            for idx, (row, vector) in enumerate(zip(rows, embedded))
        )
    else:
        model = _colbert_encoder(model_name, encode_device=encode_device)
        probe = next(model.embed([texts[0]], batch_size=1))
        dims = len(probe[0])
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "colbert": models.VectorParams(
                    size=dims,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
            },
        )
        vectors = model.embed(texts, batch_size=batch_size)
        points = (
            models.PointStruct(id=idx, payload=_clean_payload(row), vector={"colbert": vector.tolist()})
            for idx, (row, vector) in enumerate(zip(rows, vectors))
        )

    client.upload_points(collection_name=collection, points=points, batch_size=batch_size, wait=True)
    meta = {
        "index_id": index_id,
        "kind": kind,
        "model": model_name,
        "encode_backend": sparse_backend or "fastembed",
        "encode_device": resolved_device,
        "collection": collection,
        "qdrant_url": os.getenv("GRAPH_RAG_QDRANT_URL") or None,
        "base_profile": base_profile,
        "corpus_fingerprint": fingerprint,
        "chunks": len(rows),
        "batch_size": batch_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": round(time.monotonic() - started, 2),
        "memory_start": start_memory.to_dict(),
        "memory_end": memory_snapshot().to_dict(),
        "platform": platform.platform(),
    }
    (index_dir / "index_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return index_dir


@lru_cache(maxsize=8)
def _sparse_encoder_cached(model_name: str, encode_device: EncodeDevice | None):
    return _sparse_encoder(model_name, encode_device=encode_device)


@lru_cache(maxsize=8)
def _colbert_encoder_cached(model_name: str, encode_device: EncodeDevice | None):
    return _colbert_encoder(model_name, encode_device=encode_device)


def _query_encode_device(meta_device: EncodeDevice | None) -> EncodeDevice | None:
    """Device for encoding the *query* at search time. The doc index is already
    built, so query encoding is a single short forward pass — set
    ``GRAPH_RAG_EXPERIMENTAL_QUERY_DEVICE=cpu`` to keep it off a GPU that's full
    with the 4B dense model (the experimental encoders won't fit alongside it)."""
    override = os.getenv("GRAPH_RAG_EXPERIMENTAL_QUERY_DEVICE", "").strip()
    return override or meta_device  # type: ignore[return-value]


def search_experimental(
    index_dir: Path,
    query: str,
    *,
    limit: int,
    expected_kind: str | None = None,
) -> list[dict[str, Any]]:
    QdrantClient, _models, _SparseTextEmbedding, _LateInteractionTextEmbedding = _qdrant()
    meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
    if expected_kind and meta["kind"] != expected_kind:
        raise ValueError(
            f"Strategy requires {expected_kind!r} index, got {meta['kind']!r}: {index_dir}"
        )
    client, collection = _connect(QdrantClient, index_dir, meta["index_id"], meta=meta)
    encode_device = _query_encode_device(meta.get("encode_device"))
    if meta["kind"] == "splade":
        backend = meta.get("encode_backend") or _sparse_backend(meta["model"])
        if backend == "pytorch_sparse":
            vector = _sparse_encoder_cached(meta["model"], encode_device).query_embed(query)
        else:
            vector = next(_sparse_encoder_cached(meta["model"], encode_device).query_embed(query))
        query_vector = (
            "splade",
            _models.SparseVector(
                indices=vector.indices.tolist() if hasattr(vector.indices, "tolist") else list(vector.indices),
                values=vector.values.tolist() if hasattr(vector.values, "tolist") else list(vector.values),
            ),
        )
    else:
        vector = next(_colbert_encoder_cached(meta["model"], encode_device).query_embed(query))
        query_vector = ("colbert", vector)
    name, value = query_vector
    response = client.query_points(
        collection_name=collection,
        query=value,
        using=name,
        limit=limit,
        with_payload=True,
    )
    return [
        {**(point.payload or {}), "score": float(point.score), "experimental_index": meta["index_id"]}
        for point in response.points
    ]


@click.command("build-retrieval-index")
@click.option("--base-profile", required=True)
@click.option("--kind", type=click.Choice(INDEX_KINDS), required=True)
@click.option("--model", "model_name", default=None)
@click.option("--batch-size", default=4, show_default=True, type=click.IntRange(min=1, max=32))
@click.option("--min-available-gb", default=8.0, show_default=True, type=click.FloatRange(min=1))
@click.option(
    "--encode-device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
    help="ONNX/PyTorch encode device (auto picks CUDA when available).",
)
def build_retrieval_index_cmd(
    base_profile: str,
    kind: IndexKind,
    model_name: str | None,
    batch_size: int,
    min_available_gb: float,
    encode_device: EncodeDevice,
) -> None:
    """Build an immutable learned-sparse or late-interaction index."""
    path = build_experimental_index(
        base_profile=base_profile,
        kind=kind,
        model_name=model_name,
        batch_size=batch_size,
        min_available_gb=min_available_gb,
        encode_device=encode_device,
    )
    click.echo(f"Wrote {path}")
