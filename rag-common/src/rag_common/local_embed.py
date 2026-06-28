from __future__ import annotations

import gc
import logging
import os
import platform
import time
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from rag_common.config import (
    LOCAL_MODEL_DIMS,
    EmbedConfig,
    EmbedStats,
    LocalEmbedMode,
    use_cuda_bnb_4bit,
    use_mlx_q4_embed,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger("rag_common.local")

LOCAL_BATCH_SIZE = 16
# Log + yield progress for long runs (non-TTY pipes hide tqdm updates).
LOCAL_PROGRESS_CHUNKS = 256
# Cap texts per encode() call on MPS to avoid unified-memory OOM on long chunks.
MPS_ENCODE_CHUNK = 24
MPS_BATCH_SIZE = 4
MAX_EMBED_CHARS = 3000

_model_cache: dict[tuple[str, str, bool], SentenceTransformer] = {}


def _mps_encode_chunk() -> int:
    return int(os.getenv("RAG_MPS_ENCODE_CHUNK", str(MPS_ENCODE_CHUNK)))


def _mps_batch_size() -> int:
    return int(os.getenv("RAG_MPS_BATCH_SIZE", str(MPS_BATCH_SIZE)))


def _cuda_batch_size() -> int:
    return int(os.getenv("RAG_CUDA_BATCH_SIZE", str(LOCAL_BATCH_SIZE)))


def _release_mps_memory() -> None:
    """PyTorch MPS can leak unified memory across encode() calls; flush between chunks."""
    try:
        import torch

        if torch.backends.mps.is_available():
            gc.collect()
            torch.mps.empty_cache()
    except Exception:
        pass


def _model_family(model_name: str) -> str:
    lower = model_name.lower()
    if "qwen" in lower and "embedding" in lower:
        return "qwen3"
    if "jina-embeddings" in lower:
        return "jina"
    if "nomic-embed" in lower:
        return "nomic"
    if "bge-m3" in lower or "bge-m" in lower:
        return "bge-m3"
    if "bge" in lower:
        return "bge"
    return "generic"


def _jina_task(mode: LocalEmbedMode) -> str:
    """jina-embeddings-v3 selects its retrieval LoRA adapter via the `task` kwarg
    (text prefixes are NOT used — the adapter handles query/passage asymmetry)."""
    return "retrieval.query" if mode == "query" else "retrieval.passage"


def _prepare_texts(texts: list[str], *, model_name: str, mode: LocalEmbedMode) -> list[str]:
    if _model_family(model_name) == "nomic":
        prefix = "search_query: " if mode == "query" else "search_document: "
        return [prefix + t for t in texts]
    # jina-v3 uses task adapters (applied at encode time), not text prefixes.
    return texts


def _needs_trust_remote_code(model_name: str) -> bool:
    return _model_family(model_name) == "jina"


def _base_model_kwargs(model_name: str) -> dict:
    if _model_family(model_name) == "qwen3" and platform.system() == "Darwin":
        return {"attn_implementation": "eager"}
    return {}


def resolve_local_embed_device(model_name: str) -> str | None:
    """Pick device for local encode from RAG_LOCAL_EMBED_DEVICE or sensible defaults."""
    raw = os.getenv("RAG_LOCAL_EMBED_DEVICE", "auto").strip().lower()
    if raw == "cpu":
        return "cpu"
    if raw == "mps":
        return "mps"
    if raw == "cuda":
        return "cuda"
    if raw not in ("", "auto"):
        log.warning("unknown RAG_LOCAL_EMBED_DEVICE=%r; using auto", raw)

    if _model_family(model_name) == "qwen3" and platform.system() == "Darwin":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return None


def _local_batch_size(device: str | None) -> int:
    if device == "mps":
        return _mps_batch_size()
    if device == "cuda":
        return _cuda_batch_size()
    return LOCAL_BATCH_SIZE


def _bnb_4bit_model_kwargs() -> dict | None:
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        log.warning("bitsandbytes 4-bit requested but import failed (%s); using FP16", exc)
        return None

    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        "device_map": {"": 0},
    }


def _load_sentence_transformer(
    model_name: str,
    *,
    device: str | None,
    bnb_4bit: bool,
) -> SentenceTransformer:
    mk = _base_model_kwargs(model_name)
    trc = _needs_trust_remote_code(model_name)
    if bnb_4bit:
        bnb_kwargs = _bnb_4bit_model_kwargs()
        if bnb_kwargs is not None:
            try:
                log.info("loading local embed model %s (CUDA 4-bit)", model_name)
                return SentenceTransformer(model_name, model_kwargs={**mk, **bnb_kwargs})
            except Exception as exc:
                log.warning("4-bit load failed for %s (%s); falling back to FP16", model_name, exc)

    log.info(
        "loading local embed model %s device=%s trust_remote_code=%s",
        model_name,
        device or "default",
        trc,
    )
    st_kwargs: dict = {}
    if mk:
        st_kwargs["model_kwargs"] = mk
    if trc:
        st_kwargs["trust_remote_code"] = True
    return SentenceTransformer(model_name, device=device, **st_kwargs)


def _get_model_for_config(config: EmbedConfig) -> SentenceTransformer:
    device = resolve_local_embed_device(config.model)
    bnb_4bit = use_cuda_bnb_4bit(config) and device == "cuda"
    key = (config.model, device or "", bnb_4bit)
    cached = _model_cache.get(key)
    if cached is not None:
        return cached
    model = _load_sentence_transformer(config.model, device=device, bnb_4bit=bnb_4bit)
    _model_cache[key] = model
    return model


def embed_local_texts(
    texts: list[str],
    *,
    config: EmbedConfig,
    stats: EmbedStats | None = None,
    mode: LocalEmbedMode = "document",
) -> list[list[float]]:
    if not texts:
        return []

    if use_mlx_q4_embed(config.model, config):
        try:
            from rag_common.mlx_embed import embed_mlx_q4_texts

            return embed_mlx_q4_texts(texts, config=config, stats=stats, mode=mode)
        except ImportError as exc:
            log.warning(
                "RAG_LOCAL_EMBED_QUANT=%s but mlx-embeddings is not installed (%s); "
                "falling back to sentence-transformers",
                os.getenv("RAG_LOCAL_EMBED_QUANT"),
                exc,
            )

    model = _get_model_for_config(config)
    prepared = [
        t[:MAX_EMBED_CHARS] if len(t) > MAX_EMBED_CHARS else t
        for t in _prepare_texts(texts, model_name=config.model, mode=mode)
    ]
    family = _model_family(config.model)

    device = resolve_local_embed_device(config.model)
    batch_size = _local_batch_size(device)
    encode_chunk = _mps_encode_chunk() if device == "mps" else LOCAL_PROGRESS_CHUNKS

    log.info(
        "local embedding %d texts model=%s dims=%d mode=%s device=%s batch=%d",
        len(texts),
        config.model,
        config.dimensions,
        mode,
        device or "default",
        batch_size,
    )

    encode_kwargs: dict = {
        "batch_size": batch_size,
        "show_progress_bar": False,
        "normalize_embeddings": True,
    }
    if device and not use_cuda_bnb_4bit(config):
        encode_kwargs["device"] = device
    if family == "qwen3" and mode == "query":
        encode_kwargs["prompt_name"] = "query"
    if family == "qwen3":
        native = LOCAL_MODEL_DIMS.get(config.model, config.dimensions)
        if 0 < config.dimensions < native:
            encode_kwargs["truncate_dim"] = config.dimensions
    if family == "jina":
        encode_kwargs["task"] = _jina_task(mode)
        native = LOCAL_MODEL_DIMS.get(config.model, config.dimensions)
        if 0 < config.dimensions < native:
            encode_kwargs["truncate_dim"] = config.dimensions

    all_vectors: list[list[float]] = []
    total = len(prepared)
    t0 = time.monotonic()
    t_step = t0
    for start in range(0, total, encode_chunk):
        end = min(total, start + encode_chunk)
        chunk_vecs = model.encode(prepared[start:end], **encode_kwargs)
        all_vectors.extend(v.tolist() for v in chunk_vecs)
        if device == "mps":
            _release_mps_memory()
        now = time.monotonic()
        elapsed = now - t0
        rate = end / elapsed if elapsed else 0.0
        eta = (total - end) / rate if rate else 0.0
        log.info(
            "embed progress: %d/%d texts (%.1f%%, +%.1fs, total %.1fs, %.2f texts/s, eta %.1fs)",
            end,
            total,
            end / total * 100,
            now - t_step,
            elapsed,
            rate,
            eta,
        )
        t_step = now

    if total <= encode_chunk:
        if stats is not None:
            stats.add(tokens=len(texts), requests=1)
        return all_vectors

    if stats is not None:
        stats.add(tokens=len(texts), requests=max(1, (total + encode_chunk - 1) // encode_chunk))

    return all_vectors


# --------------------------------------------------------------------------
# Late chunking (additive; used only by latechunk-named profiles).
#
# Real (Jina-style) late chunking: embed a long window once to get token-level
# embeddings that carry cross-chunk context, then mean-pool each chunk's token
# span. `pool_span` is the pure-math core (no model load); the math is unit-
# tested with synthetic vectors. `embed_local_texts_with_tokens` is the model
# path. NEITHER touches the existing embed_local_texts behavior.
# --------------------------------------------------------------------------


def pool_span(
    token_vectors,
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
    *,
    method: str = "mean",
    target_dim: int | None = None,
) -> list[float]:
    """Pool the token vectors whose char span intersects [char_start, char_end).

    - Skips special/padding tokens (zero-length offset spans, e.g. (0, 0)).
    - Falls back to all real tokens if the span selects none (defensive).
    - Matryoshka-truncates to ``target_dim`` when it is smaller than native.
    - L2-normalizes (guarding against a zero vector).

    Pure NumPy; importable and testable without loading any model.
    """
    import numpy as np

    arr = np.asarray(token_vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    n = min(arr.shape[0], len(offsets))
    arr = arr[:n]
    offs = offsets[:n]

    real = [i for i, (s, e) in enumerate(offs) if e > s]
    selected = [i for i in real if offs[i][0] < char_end and offs[i][1] > char_start]
    if not selected:
        selected = real or list(range(n))

    sub = arr[selected]
    if method == "max":
        pooled = sub.max(axis=0)
    else:  # mean (default)
        pooled = sub.mean(axis=0)

    if target_dim is not None and 0 < target_dim < pooled.shape[0]:
        pooled = pooled[:target_dim]

    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32).tolist()


def embed_local_texts_with_tokens(
    texts: list[str],
    *,
    config: EmbedConfig,
    mode: LocalEmbedMode = "document",
) -> list[tuple[list[list[float]], list[tuple[int, int]]]]:
    """Return per-text (token_embeddings, char_offset_spans) for late chunking.

    Token embeddings come from SentenceTransformer.encode(output_value=
    "token_embeddings"); char offsets come from the fast tokenizer with
    return_offsets_mapping=True (special tokens get (0, 0) and are skipped by
    pool_span). Lengths are aligned defensively to the shorter of the two.
    """
    if not texts:
        return []

    model = _get_model_for_config(config)
    # CRITICAL for real late chunking: do NOT clamp to MAX_EMBED_CHARS (3000) here —
    # that would shrink a 20k-char window down to ~one chunk and defeat the whole
    # point (cross-chunk context). Cap generously at the model's token budget
    # (chars ≈ max_seq_length * ~5) and let the tokenizer truncate precisely.
    max_len = getattr(model, "max_seq_length", None) or 512
    window_char_cap = max(MAX_EMBED_CHARS, max_len * 5)
    prepared = [
        t[:window_char_cap] if len(t) > window_char_cap else t
        for t in _prepare_texts(texts, model_name=config.model, mode=mode)
    ]
    device = resolve_local_embed_device(config.model)
    batch_size = _local_batch_size(device)

    encode_kwargs: dict = {
        "output_value": "token_embeddings",
        "batch_size": batch_size,
        "show_progress_bar": False,
    }
    if _model_family(config.model) == "jina":
        encode_kwargs["task"] = _jina_task(mode)
    tok_emb = model.encode(prepared, **encode_kwargs)
    out: list[tuple[list[list[float]], list[tuple[int, int]]]] = []
    for text, emb in zip(prepared, tok_emb):
        try:
            enc = model.tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_len,
            )
            offsets = [tuple(o) for o in enc.get("offset_mapping", [])]
        except Exception as exc:  # noqa: BLE001 — degrade to no offsets
            log.warning("offset mapping failed (%s); chunk spans will use full window", exc)
            offsets = []
        try:
            vecs = emb.tolist()
        except AttributeError:
            vecs = [list(v) for v in emb]
        if offsets and len(offsets) != len(vecs):
            n = min(len(offsets), len(vecs))
            vecs, offsets = vecs[:n], offsets[:n]
        out.append((vecs, offsets))
    return out
