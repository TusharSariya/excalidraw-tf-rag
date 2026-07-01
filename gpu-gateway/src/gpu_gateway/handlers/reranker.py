"""Cross-encoder reranker deployment (bge-reranker-v2-m3).

user_config delivered via reconfigure() — model path stored there, loaded lazily.
Eviction/park mechanics live in the shared _EvictableModel mixin (no bnb here, so
the base validate hook is fine — a cross-encoder survives device moves cleanly).
"""

from __future__ import annotations

import logging

from ray import serve

from gpu_gateway.handlers.evictable import _EvictableModel

log = logging.getLogger("gpu_gateway.handlers.reranker")


@serve.deployment
class RerankerDeployment(_EvictableModel):
    def __init__(self) -> None:
        self._model = None
        self._model_path: str = ""

    def reconfigure(self, config: dict) -> None:
        # Store model path; load lazily on first __call__ to avoid startup OOM.
        self._model_path = config["model"]
        self._model = None  # reset so __call__ reloads when config changes
        self._parked = False
        self._configure_evict(config)

    def _load_from_disk(self) -> None:
        from sentence_transformers import CrossEncoder

        log.info("lazy-loading reranker %s", self._model_path)
        self._model = CrossEncoder(self._model_path, device="cuda")
        log.info("loaded %s", self._model_path)

    async def __call__(self, query: str, passages: list[str]) -> list[float]:
        self._ensure_loaded()
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, batch_size=8)
        return [float(s) for s in scores]
