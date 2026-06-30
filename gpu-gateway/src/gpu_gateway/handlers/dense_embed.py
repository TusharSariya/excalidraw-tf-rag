"""Dense embedding deployments (Qwen3-4B and Qwen3-0.6B).

user_config is delivered via reconfigure() in Ray Serve 2.x — NOT __init__.
Models are loaded LAZILY on the first __call__ to avoid all three models loading
into VRAM simultaneously at startup (which would OOM an 8 GB card before the
VRAMCoordinator lock is active). After the first load the model stays in VRAM.
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
            log.info("loaded %s", self._model_path)

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
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
            log.info("loaded %s", self._model_path)


@serve.deployment
class DenseEmbed06B(_DenseEmbedBase):
    pass
