"""ProcessPoolExecutor shutdown helpers that reap worker processes on interrupt."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def _worker_processes(executor: ProcessPoolExecutor) -> list[Any]:
    processes = getattr(executor, "_processes", None)
    if not processes:
        return []
    return list(processes.values())


def shutdown_process_pool(
    executor: ProcessPoolExecutor,
    *,
    force: bool = False,
    timeout_s: float = 15.0,
    log: logging.Logger | None = None,
) -> None:
    """Cancel pending work, shutdown the pool, and terminate/kill straggler workers."""
    logger = log or logging.getLogger(__name__)
    try:
        executor.shutdown(wait=not force, cancel_futures=True)
    except Exception as exc:  # noqa: BLE001 - best-effort cleanup
        logger.warning("process pool shutdown raised %s; continuing worker cleanup", exc)

    if not force:
        return

    workers = _worker_processes(executor)
    if not workers:
        return

    terminated = 0
    for process in workers:
        if process.is_alive():
            process.terminate()
            terminated += 1

    killed = 0
    for process in workers:
        process.join(timeout=timeout_s)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
            killed += 1

    if terminated or killed:
        logger.warning(
            "force-terminated %d process-pool worker(s); killed %d straggler(s)",
            terminated,
            killed,
        )
