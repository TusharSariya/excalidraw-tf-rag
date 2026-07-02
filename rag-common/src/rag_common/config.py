from __future__ import annotations

import os
import platform
import threading
from dataclasses import dataclass, field
from typing import Literal


def local_embed_quant(config: EmbedConfig | None = None) -> str | None:
    """Quantization mode: profile quant first, then RAG_LOCAL_EMBED_QUANT env."""
    if config is not None and config.quant:
        raw = config.quant.strip().lower()
    else:
        raw = os.getenv("RAG_LOCAL_EMBED_QUANT", "").strip().lower()
    if raw in ("", "none", "fp16", "bf16", "off", "false", "0"):
        return None
    return raw


def use_cuda_bnb_4bit(config: EmbedConfig | None = None) -> bool:
    """True when profile/env requests 4-bit quant on a CUDA host (bitsandbytes path)."""
    quant = local_embed_quant(config)
    if quant not in ("4bit", "q4"):
        return False
    if platform.system() == "Darwin":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolved_embed_backend(
    config: EmbedConfig | None = None, *, model_name: str | None = None
) -> str:
    """Concrete embed backend AS RESOLVED ON THIS HOST.

    The profile ``backend`` field is "local" for every Qwen rung regardless of whether
    bitsandbytes (CUDA) quantization actually runs, so multiple quant stacks can share
    a profile name. This returns a host-specific tag (``cuda-bnb-4bit`` vs plain
    ``local-fp``) so a query host can be matched against the host that BUILT an index.
    """
    if use_cuda_bnb_4bit(config):
        return "cuda-bnb-4bit"
    backend = config.backend if config else "unknown"
    quant = local_embed_quant(config)
    if backend == "local":
        return f"local-{quant}" if quant else "local-fp"
    return backend


EmbedBackend = Literal["local"]
LocalEmbedMode = Literal["query", "document"]

LOCAL_DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LOCAL_DEFAULT_DIMS = 1024

LOCAL_MODEL_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "nomic-ai/nomic-embed-text-v2-moe": 768,
    "jinaai/jina-embeddings-v3": 1024,
}


def local_model_dimensions(model_name: str) -> int:
    for key in ("RAG_LOCAL_EMBED_DIMS", "GRAPH_RAG_LOCAL_EMBED_DIMS"):
        value = os.getenv(key)
        if value:
            return int(value)
    return LOCAL_MODEL_DIMS.get(model_name, LOCAL_DEFAULT_DIMS)


@dataclass(frozen=True)
class EmbedConfig:
    backend: EmbedBackend
    model: str
    dimensions: int
    profile: str | None = None
    quant: str | None = None

    @property
    def is_remote(self) -> bool:
        return False


@dataclass
class EmbedStats:
    tokens: int = 0
    requests: int = 0
    effective_config: EmbedConfig | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, *, tokens: int, requests: int = 1) -> None:
        with self._lock:
            self.tokens += tokens
            self.requests += requests

    def set_effective_config(self, config: EmbedConfig) -> None:
        with self._lock:
            self.effective_config = config


def local_config(*, model: str | None = None, profile: str | None = None) -> EmbedConfig:
    name = model or os.getenv("RAG_LOCAL_EMBED_MODEL", LOCAL_DEFAULT_MODEL)
    quant = os.getenv("RAG_LOCAL_EMBED_QUANT", "").strip().lower() or None
    if quant in ("", "none", "off", "false", "0"):
        quant = None
    return EmbedConfig(
        backend="local",
        model=name,
        dimensions=local_model_dimensions(name),
        profile=profile,
        quant=quant,
    )


def embed_cost_per_million(backend: EmbedBackend) -> float:
    return 0.0
