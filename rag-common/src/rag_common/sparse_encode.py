"""Learned-sparse encoders (shared by graph-layout-rag and rag-literature-rag).

A *learned sparse* retriever maps text to a small ``{token_id: weight}`` bag where
the weights are predicted by a model (impact weighting / lexical expansion) rather
than by a BM25 term-frequency formula. Two implementations live here, both behind
the :class:`SparseEncoder` Protocol so the ingest/retrieval plumbing is
model-agnostic:

* :class:`BGEM3SparseEncoder` — the sparse (lexical) head of ``BAAI/bge-m3``. This
  is the default: bge-m3 is a popular multi-functional model, so one model can
  later serve both the dense and the sparse channel. Requires the optional
  ``FlagEmbedding`` dependency — install with ``uv sync --extra sparse`` in the
  tool package (the extra is declared on ``rag-common``).
* :class:`SpladeV3Encoder` — ``naver/splade-v3`` (and siblings), reusing the
  sentence-transformers ``SparseEncoder`` path already exercised by the eval
  bake-off. sentence-transformers is a core ``rag-common`` dependency, but the
  ``SparseEncoder`` class needs a recent release (>= 5.0); the ``sparse`` extra
  is the supported way to pull a compatible stack.

Both classes are lazy: constructing one loads **no** model weights. The model is
only materialised on the first ``encode_documents`` / ``encode_query`` call, so
importing this module (and instantiating an encoder to read its ``model_id``) is
free and import-safe on a CPU-only / offline host.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

BGE_M3_MODEL_ID = "BAAI/bge-m3"
SPLADE_V3_MODEL_ID = "naver/splade-v3"


@runtime_checkable
class SparseEncoder(Protocol):
    """Pluggable learned-sparse encoder.

    ``model_id`` / ``version`` identify the concrete model + weighting scheme and
    are persisted in the sparse index metadata so a query-time encoder that does
    not match the index the vectors were built with is refused (HG1-style guard).
    """

    @property
    def model_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    def encode_documents(self, texts: list[str]) -> list[dict[int, float]]:
        """Encode a batch of documents to ``{token_id: weight}`` bags (one per text)."""
        ...

    def encode_query(self, text: str) -> dict[int, float]:
        """Encode a single query to a ``{token_id: weight}`` bag."""
        ...


def _prune(weights: dict[int, float]) -> dict[int, float]:
    """Drop non-positive weights so the inverted index only stores real impact."""
    return {int(tid): float(w) for tid, w in weights.items() if float(w) > 0.0}


class BGEM3SparseEncoder:
    """``BAAI/bge-m3`` sparse (lexical) head via the optional ``FlagEmbedding`` dep."""

    def __init__(
        self,
        model_id: str = BGE_M3_MODEL_ID,
        *,
        device: str | None = None,
        use_fp16: bool = True,
        version: str = "1",
    ) -> None:
        self._model_id = model_id
        self._version = version
        self._device = device
        self._use_fp16 = use_fp16
        self._model = None  # lazy — no weights loaded at construction

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def version(self) -> str:
        return f"bgem3-{self._version}"

    def _ensure_model(self):  # noqa: ANN202 - external model type
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "BGEM3SparseEncoder needs FlagEmbedding. Install the optional "
                    "sparse-retrieval dependency: uv sync --extra sparse"
                ) from exc
            kwargs: dict[str, object] = {"use_fp16": self._use_fp16}
            if self._device:
                kwargs["devices"] = self._device
            self._model = BGEM3FlagModel(self._model_id, **kwargs)
        return self._model

    def encode_documents(self, texts: list[str]) -> list[dict[int, float]]:
        model = self._ensure_model()
        out = model.encode(
            list(texts),
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [_prune(w) for w in out["lexical_weights"]]

    def encode_query(self, text: str) -> dict[int, float]:
        model = self._ensure_model()
        out = model.encode(
            [text],
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return _prune(out["lexical_weights"][0])


class SpladeV3Encoder:
    """``naver/splade-v3`` via sentence-transformers ``SparseEncoder``.

    Uses the model's asymmetric surface — ``encode_document`` (full MLM expansion)
    on the doc side and ``encode_query`` (tokenizer-side, no forward pass for the
    inference-free variants) on the query side — mirroring
    ``eval/splade_v3_encoder.py``.
    """

    def __init__(
        self,
        model_id: str = SPLADE_V3_MODEL_ID,
        *,
        device: str | None = None,
        batch_size: int = 16,
        version: str = "1",
    ) -> None:
        self._model_id = model_id
        self._version = version
        self._device = device
        self._batch_size = batch_size
        self._model = None  # lazy

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def version(self) -> str:
        return f"splade-{self._version}"

    def _ensure_model(self):  # noqa: ANN202 - external model type
        if self._model is None:
            try:
                from sentence_transformers import SparseEncoder
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "SpladeV3Encoder needs sentence-transformers>=5.0 (SparseEncoder). "
                    "Install the optional sparse-retrieval dependency: uv sync --extra sparse"
                ) from exc
            device = self._device or _resolve_torch_device()
            self._model = SparseEncoder(self._model_id, device=device)
        return self._model

    def encode_documents(self, texts: list[str]) -> list[dict[int, float]]:
        model = self._ensure_model()
        embeddings = model.encode_document(
            list(texts),
            batch_size=self._batch_size,
            convert_to_tensor=False,
            convert_to_sparse_tensor=True,
            show_progress_bar=False,
        )
        return [_sparse_tensor_to_dict(t) for t in embeddings]

    def encode_query(self, text: str) -> dict[int, float]:
        model = self._ensure_model()
        embeddings = model.encode_query(
            [text],
            batch_size=1,
            convert_to_tensor=False,
            convert_to_sparse_tensor=True,
            show_progress_bar=False,
        )
        return _sparse_tensor_to_dict(embeddings[0])


def _resolve_torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch missing/broken
        pass
    return "cpu"


def _sparse_tensor_to_dict(tensor) -> dict[int, float]:  # noqa: ANN001 - torch.Tensor
    import torch

    if not isinstance(tensor, torch.Tensor):  # pragma: no cover - defensive
        raise TypeError(f"expected sparse torch.Tensor, got {type(tensor)!r}")
    coalesced = tensor.coalesce()
    idx = coalesced.indices()
    indices = idx[0] if coalesced.dim() == 1 else idx[1]
    values = coalesced.values()
    out: dict[int, float] = {}
    for token_id, weight in zip(indices.cpu().tolist(), values.cpu().tolist()):
        w = float(weight)
        if w > 0.0:
            out[int(token_id)] = w
    return out


def resolve_sparse_encoder(
    *,
    encoder: str = "bgem3",
    model_id: str | None = None,
    device: str | None = None,
    version: str = "1",
) -> SparseEncoder:
    """Instantiate a sparse encoder by short name. Loads no weights (lazy)."""
    kind = (encoder or "bgem3").strip().lower()
    if kind in {"bgem3", "bge-m3", "bge_m3", "bge"}:
        return BGEM3SparseEncoder(model_id or BGE_M3_MODEL_ID, device=device, version=version)
    if kind in {"splade", "splade-v3", "spladev3", "splade_v3"}:
        return SpladeV3Encoder(model_id or SPLADE_V3_MODEL_ID, device=device, version=version)
    raise ValueError(f"Unknown sparse encoder {encoder!r}; choose 'bgem3' (default) or 'splade'.")


__all__ = [
    "BGE_M3_MODEL_ID",
    "SPLADE_V3_MODEL_ID",
    "BGEM3SparseEncoder",
    "SpladeV3Encoder",
    "SparseEncoder",
    "resolve_sparse_encoder",
]
