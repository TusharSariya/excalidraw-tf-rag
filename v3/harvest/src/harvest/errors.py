"""Shared harvest worker / source exceptions."""

from __future__ import annotations


class RateLimited(Exception):
    """Upstream returned HTTP 429 — stop draining so pgmq VT spaces retries.

    Sources raise this (or a plain HTTP 429 that the worker maps to this) instead
    of continuing to the next queue message. Chewing the backlog while rate-limited
    only burns ``read_ct`` toward the poison-pill cap and deepens the ban.
    """

    def __init__(self, source: str, *, retry_after: str | None = None) -> None:
        self.source = source
        self.retry_after = retry_after
        detail = f"{source} rate-limited"
        if retry_after:
            detail = f"{detail} (retry_after={retry_after})"
        super().__init__(detail)
