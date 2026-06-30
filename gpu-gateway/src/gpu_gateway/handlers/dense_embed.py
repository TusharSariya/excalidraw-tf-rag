"""Dense embedding deployments (Qwen3-4B and Qwen3-0.6B).

user_config is delivered via reconfigure() in Ray Serve 2.x — NOT __init__.
Weight loading must happen in reconfigure() or the model will never load.
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
        from sentence_transformers import SentenceTransformer

        path = config["model"]
        if self._model is None or path != self._model_path:
            log.info("loading dense embed model %s", path)
            self._model = SentenceTransformer(path, device="cuda")
            self._model_path = path
            log.info("loaded %s", path)

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("model not loaded — reconfigure() not called yet")
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()


@serve.deployment
class DenseEmbed4B(_DenseEmbedBase):
    pass


@serve.deployment
class DenseEmbed06B(_DenseEmbedBase):
    pass
