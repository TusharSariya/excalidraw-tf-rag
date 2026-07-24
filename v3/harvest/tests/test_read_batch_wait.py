"""Unit tests for SupabasePersistence.read_batch wait / empty-exit pacing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from harvest.persistence.supabase import SupabasePersistence


def _store_with_lengths(lengths: list[int]) -> SupabasePersistence:
    """Persistence whose queue_visible_length yields ``lengths`` then repeats the last."""
    store = SupabasePersistence(supabase=MagicMock())
    calls = {"i": 0}

    def visible(_queue: str) -> int:
        i = calls["i"]
        calls["i"] += 1
        return lengths[i] if i < len(lengths) else lengths[-1]

    store.queue_visible_length = visible  # type: ignore[method-assign]

    rpc = MagicMock()
    rpc.execute.return_value = MagicMock(data=[])
    store.supabase.schema.return_value.rpc.return_value = rpc
    return store


def test_read_batch_short_circuits_when_queue_already_full() -> None:
    store = _store_with_lengths([100])
    sleeps: list[float] = []

    with patch("harvest.persistence.supabase.time.sleep", side_effect=sleeps.append):
        rows = store.read_batch(
            "harvest_arxiv",
            n=100,
            vt_seconds=120,
            wait_seconds=120.0,
            empty_exit_seconds=10.0,
        )

    assert rows == []
    assert sleeps == []


def test_read_batch_empty_exit_and_poll_backoff() -> None:
    """Continuous zeros: fast 0.1s polls for 1s, then 1s polls, exit by ~10s empty."""
    store = _store_with_lengths([0])
    sleeps: list[float] = []
    clock = {"ns": 0}

    def time_ns() -> int:
        return clock["ns"]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["ns"] += int(seconds * 1e9)

    with (
        patch("harvest.persistence.supabase.time.time_ns", side_effect=time_ns),
        patch("harvest.persistence.supabase.time.sleep", side_effect=sleep),
    ):
        rows = store.read_batch(
            "harvest_arxiv",
            n=100,
            vt_seconds=120,
            wait_seconds=120.0,
            empty_exit_seconds=10.0,
            poll_fast_seconds=0.1,
            poll_fast_for_seconds=1.0,
            poll_slow_seconds=1.0,
        )

    assert rows == []
    assert sleeps
    assert all(s in (0.1, 1.0) for s in sleeps)
    assert 0.1 in sleeps
    assert 1.0 in sleeps
    assert sum(sleeps) <= 11.0


def test_read_batch_resets_empty_timer_when_messages_appear() -> None:
    """Seeing length > 0 resets empty-exit; subsequent zeros need a fresh 10s."""
    # ~0.5s of zeros, a partial fill, then zeros again — must not empty-exit
    # until 10s of continuous zero *after* the partial fill.
    lengths = [0, 0, 0, 0, 0, 5, 0]
    store = _store_with_lengths(lengths)
    sleeps: list[float] = []
    clock = {"ns": 0}
    seen: list[int] = []

    orig_visible = store.queue_visible_length

    def visible(queue: str) -> int:
        v = orig_visible(queue)
        seen.append(v)
        return v

    store.queue_visible_length = visible  # type: ignore[method-assign]

    def time_ns() -> int:
        return clock["ns"]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["ns"] += int(seconds * 1e9)

    with (
        patch("harvest.persistence.supabase.time.time_ns", side_effect=time_ns),
        patch("harvest.persistence.supabase.time.sleep", side_effect=sleep),
    ):
        store.read_batch(
            "harvest_arxiv",
            n=100,
            vt_seconds=120,
            wait_seconds=120.0,
            empty_exit_seconds=10.0,
            poll_fast_seconds=0.1,
            poll_fast_for_seconds=1.0,
            poll_slow_seconds=1.0,
        )

    assert 5 in seen
    # After the 5, need another ~10s of zeros → total sleep well above 10s.
    assert sum(sleeps) >= 10.0
