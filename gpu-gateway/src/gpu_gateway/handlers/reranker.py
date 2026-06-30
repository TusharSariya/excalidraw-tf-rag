"""Cross-encoder reranker deployment (bge-reranker-v2-m3).

user_config delivered via reconfigure() — model loaded there, not __init__.
"""

from __future__ import annotations

import logging

from ray import serve

log = logging.getLogger("gpu_gateway.handlers.reranker")


@serve.deployment
class RerankerDeployment:
    def __init__(self) -> None:
        self._model = None
        self._model_path: str = ""

    def reconfigure(self, config: dict) -> None:
        # Store model path; load lazily on first __call__ to avoid startup OOM.
        self._model_path = config["model"]
        self._model = None  # reset so __call__ reloads when config changes

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            log.info("lazy-loading reranker %s", self._model_path)
            self._model = CrossEncoder(self._model_path, device="cuda")
            log.info("loaded %s", self._model_path)

    async def unload(self) -> None:
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            log.info("unloaded %s", self._model_path)

    async def __call__(self, query: str, passages: list[str]) -> list[float]:
        self._ensure_loaded()
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, batch_size=8)
        return [float(s) for s in scores]
