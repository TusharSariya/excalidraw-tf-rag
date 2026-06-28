from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

from rag_common.process_pool import shutdown_process_pool


class _FakeProcess:
    def __init__(self, *, alive_after_terminate: bool = False) -> None:
        self.alive = True
        self.alive_after_terminate = alive_after_terminate
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls = 0

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.alive_after_terminate:
            self.alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1
        if not self.alive_after_terminate:
            self.alive = False


class _FakeExecutor:
    def __init__(self, workers: list[_FakeProcess]) -> None:
        self._processes = {id(worker): worker for worker in workers}
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


def test_shutdown_process_pool_graceful_wait():
    executor = _FakeExecutor([])
    shutdown_process_pool(executor, force=False)  # type: ignore[arg-type]
    assert executor.shutdown_calls == [(True, True)]


def test_shutdown_process_pool_force_terminates_workers(caplog):
    workers = [_FakeProcess(), _FakeProcess(alive_after_terminate=True)]
    executor = _FakeExecutor(workers)
    with caplog.at_level(logging.WARNING):
        shutdown_process_pool(
            executor,  # type: ignore[arg-type]
            force=True,
            timeout_s=0.01,
            log=logging.getLogger("test"),
        )
    assert executor.shutdown_calls == [(False, True)]
    assert workers[0].terminate_calls == 1
    assert workers[0].kill_calls == 0
    assert workers[1].terminate_calls == 1
    assert workers[1].kill_calls == 1
    assert "force-terminated 2 process-pool worker(s)" in caplog.text


def test_shutdown_process_pool_real_executor_no_workers():
    with ProcessPoolExecutor(max_workers=1) as executor:
        pass
    shutdown_process_pool(executor, force=True)
