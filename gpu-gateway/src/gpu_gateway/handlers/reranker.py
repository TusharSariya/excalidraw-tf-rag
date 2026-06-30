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
        from sentence_transformers import CrossEncoder

        path = config["model"]
        if self._model is None or path != self._model_path:
            log.info("loading reranker %s", path)
            self._model = CrossEncoder(path, device="cuda")
            self._model_path = path
            log.info("loaded %s", path)

    async def __call__(self, query: str, passages: list[str]) -> list[float]:
        if self._model is None:
            raise RuntimeError("reranker not loaded — reconfigure() not called yet")
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, batch_size=8)
        return [float(s) for s in scores]
