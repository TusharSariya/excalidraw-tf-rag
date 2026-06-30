"""VRAMCoordinator — single-replica Ray actor that serializes LLM vs embed/rerank requests.

LLM (Ollama, ~4.0GB) and embed+rerank (up to 4.0GB) cannot run simultaneously within
the 6.5GB usable VRAM budget. This actor holds a single asyncio.Lock so only one
VRAM class runs at a time, across all Ray actor processes.

NOTE: The lock lives inside a single-replica Ray actor, making it a genuine
cross-process mutex — unlike asyncio.Semaphore, which is only in-process.
"""

from __future__ import annotations

import asyncio

import ray


@ray.remote(num_cpus=0)
class VRAMCoordinator:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def acquire(self, kind: str) -> None:  # noqa: ARG002 — kind reserved for logging
        await self._lock.acquire()

    async def release(self, kind: str) -> None:  # noqa: ARG002
        self._lock.release()
