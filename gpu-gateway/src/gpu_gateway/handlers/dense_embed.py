"""Dense embedding deployments (Qwen3-4B and Qwen3-0.6B).

user_config is delivered via reconfigure() in Ray Serve 2.x — NOT __init__.
Models are loaded LAZILY on first __call__ and EVICTED before LLM requests
(GatewayHTTP calls unload() before forwarding to Ollama so the GPU is free).
"""

from __future__ import annotations

import logging

from ray import serve

log = logging.getLogger("gpu_gateway.handlers.dense_embed")


class _DenseEmbedBase:
    """Shared implementation; subclassed for 4B and 0.6B deployments."""

    def __init__(self) -> None:
        self._model = None
        self._model_path: str = ""

    def reconfigure(self, config: dict) -> None:
        # Store the model path; actual loading is deferred to the first __call__.
        self._model_path = config["model"]
        self._model = None  # reset so __call__ reloads when config changes

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("lazy-loading dense embed model %s", self._model_path)
            self._model = SentenceTransformer(self._model_path, device="cuda")
            # Cap input sequence length — long code chunks produce O(seq²) attention
            # tensors that OOM an 8 GB card even at batch_size=32.
            self._model.max_seq_length = 2048
            log.info("loaded %s", self._model_path)

    async def unload(self) -> None:
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            log.info("unloaded %s", self._model_path)

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        # batch_size=32 keeps activation memory well under 8 GB regardless of how many
        # texts the caller sends in a single HTTP request.
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return vecs.tolist()


@serve.deployment
class DenseEmbed4B(_DenseEmbedBase):
    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from transformers import BitsAndBytesConfig
            # Qwen3-4B is ~8 GB in bf16 — OOMs on an 8 GB card.
            # BitsAndBytesConfig(load_in_8bit=True) reduces to ~4 GB.
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            log.info("lazy-loading dense embed model %s (int8, 1024-dim MRL)", self._model_path)
            self._model = SentenceTransformer(
                self._model_path,
                device="cuda",
                truncate_dim=1024,
                model_kwargs={"quantization_config": bnb},
            )
            # 4B at int8 ≈ 5.7 GB; only ~2 GB left for activations.
            # seq=256 keeps FFN activations to <100 MB per layer.
            self._model.max_seq_length = 256
            log.info("loaded %s", self._model_path)

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        # batch_size=8, seq=256: FFN activation = 8 × 256 × 8192 × 2B ≈ 32 MB/layer —
        # well within the ~1.5 GB headroom left after loading 5.7 GB of int8 weights.
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=8)
        return vecs.tolist()


@serve.deployment
class DenseEmbed06B(_DenseEmbedBase):
    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("lazy-loading dense embed model %s", self._model_path)
            self._model = SentenceTransformer(self._model_path, device="cuda")
            # 512 tokens: attention scales as seq², and 2048 OOMs an 8 GB card.
            self._model.max_seq_length = 512
            log.info("loaded %s", self._model_path)

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        # batch_size=4: 0.6B FFN intermediate is large; batch_size=32 OOMs
        # (28 layers × 32 × 512 × 4096 × 2B ≈ 7 GB of activations alone).
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=4)
        return vecs.tolist()
