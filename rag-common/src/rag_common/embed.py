from __future__ import annotations

import logging
import os

from rag_common.config import EmbedConfig, EmbedStats, LocalEmbedMode
from rag_common.local_embed import embed_local_texts
from rag_common.resolve import resolve_embed_config

log = logging.getLogger("rag_common.embed")

__all__ = [
    "EmbedConfig",
    "EmbedStats",
    "embed_query",
    "embed_texts",
    "finalize_embed_config",
    "resolve_embed_config",
    "resolve_workers",
]


def resolve_workers(workers: int | None = None, *, prefix: str = "") -> int:
    if workers is not None:
        return max(1, workers)
    for key in (f"{prefix}WORKERS", "RAG_EMBED_WORKERS"):
        env = os.getenv(key)
        if env:
            return max(1, int(env))
    cpu = os.cpu_count() or 4
    return max(2, min(8, cpu))


def finalize_embed_config(
    cfg: EmbedConfig,
    *,
    prefix: str = "",
    allow_fallback: bool = True,
) -> EmbedConfig:
    """Local is the only embed backend, so there is nothing to probe/fall back from."""
    return cfg


def embed_texts(
    texts: list[str],
    *,
    config: EmbedConfig | None = None,
    stats: EmbedStats | None = None,
    workers: int | None = None,
    prefix: str = "",
    allow_fallback: bool = True,
    probe: bool = True,
    mode: LocalEmbedMode = "document",
    titles: list[str] | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    cfg = config or resolve_embed_config(prefix)

    if os.getenv("RAG_GPU_GATEWAY_URL"):
        from rag_common.remote_embed import embed_remote_texts

        model_id = getattr(cfg, "profile", None) or cfg.model
        return embed_remote_texts(texts, model_id=model_id)

    if stats is not None:
        stats.set_effective_config(cfg)
    return embed_local_texts(texts, config=cfg, stats=stats, mode=mode)


def embed_query(
    text: str,
    *,
    config: EmbedConfig | None = None,
    prefix: str = "",
    allow_fallback: bool = True,
    probe: bool = False,
) -> list[float]:
    return embed_texts(
        [text],
        config=config,
        prefix=prefix,
        allow_fallback=allow_fallback,
        workers=1,
        probe=probe,
        mode="query",
    )[0]
