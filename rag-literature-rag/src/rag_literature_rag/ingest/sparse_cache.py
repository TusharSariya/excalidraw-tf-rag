"""Disk cache for learned-sparse chunk encodings (mirrors ``embed_cache``).

Key: sha256 of (encoder model_id, version, encoded text) -> the ``{token_id:
weight}`` bag as JSON. A cache hit skips the (GPU-bound) sparse encoder entirely,
so a ``--force`` / resumed ingest re-encodes only chunks whose text or encoder
changed — same incrementality guarantee as the dense embed cache.

Lives at ``data/sparse_cache/`` (gitignored, like the rest of ``data/``). Written
atomically; concurrent writers for one key are safe (identical content per key).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from rag_literature_rag.paths import PKG_ROOT

log = logging.getLogger(__name__)

_CACHE_VERSION = 1
_CACHE_DIR = PKG_ROOT / "data" / "sparse_cache"


def cache_key(*, model_id: str, version: str, text: str) -> str:
    raw = f"v{_CACHE_VERSION}:{model_id}:{version}\x00{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / key[:2] / f"{key}.json"


def get_many(keys: list[str]) -> list[dict[int, float] | None]:
    return [_get_one(key) for key in keys]


def _get_one(key: str) -> dict[int, float] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {int(tid): float(w) for tid, w in data["tokens"].items()}
    except Exception as exc:  # noqa: BLE001
        log.debug("sparse_cache: corrupt entry %s — %s", path.name, exc)
        path.unlink(missing_ok=True)
        return None


def put_many(keys: list[str], token_dicts: list[dict[int, float]]) -> None:
    for key, tokens in zip(keys, token_dicts):
        _put_one(key, tokens)


def _put_one(key: str, tokens: dict[int, float]) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"v": _CACHE_VERSION, "tokens": {str(t): w for t, w in tokens.items()}})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def encode_documents_cached(texts: list[str], *, encoder) -> list[dict[int, float]]:  # noqa: ANN001
    """Encode ``texts`` to token bags, serving cache hits and encoding only misses."""
    keys = [cache_key(model_id=encoder.model_id, version=encoder.version, text=t) for t in texts]
    cached = get_many(keys)
    miss_idx = [i for i, v in enumerate(cached) if v is None]
    if miss_idx:
        encoded = encoder.encode_documents([texts[i] for i in miss_idx])
        put_many([keys[i] for i in miss_idx], encoded)
        for i, bag in zip(miss_idx, encoded):
            cached[i] = bag
    return [bag if bag is not None else {} for bag in cached]


def cache_stats() -> dict[str, int]:
    if not _CACHE_DIR.exists():
        return {"entries": 0, "size_mb": 0}
    entries = sum(1 for _ in _CACHE_DIR.rglob("*.json"))
    size = sum(f.stat().st_size for f in _CACHE_DIR.rglob("*.json"))
    return {"entries": entries, "size_mb": round(size / 1024 / 1024)}
