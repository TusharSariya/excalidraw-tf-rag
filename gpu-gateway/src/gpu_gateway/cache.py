"""SQLiteCache — Ray actor holding the aiosqlite embed/rerank cache.

aiosqlite is required (not stdlib sqlite3) because this actor's methods are
async and sqlite3 connections are thread-local — they raise ProgrammingError
when awaited across event-loop iterations.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any

import aiosqlite
import ray

log = logging.getLogger("gpu_gateway.cache")

DB_PATH = os.getenv("GPU_GATEWAY_CACHE_PATH", "/tmp/gpu_gateway_cache.db")
MAX_ROWS = 500_000
EVICT_ROWS = 50_000

MODEL_DIMS: dict[str, int] = {
    "cuda-qwen4b-1024": 1024,
    "cuda-qwen0.6b-1024": 1024,
}


def embed_cache_key(model_id: str, text: str) -> str:
    dims = MODEL_DIMS.get(model_id, 0)
    return hashlib.sha256(f"{model_id}|{dims}|embed|{text}".encode()).hexdigest()


def rerank_cache_key(model_id: str, query: str, passage: str) -> str:
    return hashlib.sha256(f"{model_id}|rerank|{query}|{passage}".encode()).hexdigest()


@ray.remote(num_cpus=0)
class SQLiteCache:
    def __init__(self) -> None:
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(DB_PATH)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS embed_cache (
                    key        TEXT NOT NULL,
                    vector     BLOB NOT NULL,
                    model_id   TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)
            await self._db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_key ON embed_cache(key)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON embed_cache(created_at)"
            )
            await self._db.commit()
        return self._db

    async def batch_get(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        db = await self._conn()
        placeholders = ",".join("?" * len(keys))
        rows = await db.execute_fetchall(
            f"SELECT key, vector FROM embed_cache WHERE key IN ({placeholders})", keys
        )
        import struct
        result: dict[str, list[float]] = {}
        for key, blob in rows:
            n = len(blob) // 4
            result[key] = list(struct.unpack(f"{n}f", blob))
        return result

    async def batch_set(self, entries: list[tuple[str, str, list[float]]]) -> None:
        """entries: [(key, model_id, vector), ...]"""
        if not entries:
            return
        import struct
        import time

        db = await self._conn()
        now = int(time.time())
        async with self._lock:
            await db.executemany(
                "INSERT OR REPLACE INTO embed_cache(key, vector, model_id, created_at) VALUES (?,?,?,?)",
                [
                    (key, model_id, struct.pack(f"{len(vec)}f", *vec), now)
                    for key, model_id, vec in entries
                ],
            )
            await db.commit()
            await self._evict_if_needed(db)

    async def _evict_if_needed(self, db: aiosqlite.Connection) -> None:
        row = await db.execute_fetchall("SELECT COUNT(*) FROM embed_cache")
        count = row[0][0] if row else 0
        if count > MAX_ROWS:
            await db.execute(
                "DELETE FROM embed_cache WHERE rowid IN "
                "(SELECT rowid FROM embed_cache ORDER BY created_at ASC LIMIT ?)",
                (EVICT_ROWS,),
            )
            await db.commit()
            log.info("evicted %d cache rows (was %d)", EVICT_ROWS, count)

    async def count(self) -> int:
        db = await self._conn()
        row = await db.execute_fetchall("SELECT COUNT(*) FROM embed_cache")
        return row[0][0] if row else 0
