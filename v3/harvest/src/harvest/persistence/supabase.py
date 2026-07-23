"""Supabase implementation of the Persistence contract.

Source-agnostic by construction: ``write`` upserts whatever rows a mapper hands
it, reading each row's ``__table__``/``__conflict__``. It imports no source and
no row type — that absence is the seam working.
"""

from __future__ import annotations

import time
from typing import Any

from supabase import Client, create_client

from ..settings import SupabaseSettings
from .base import Write
from .models import Seed

# source key (queue / --source) -> its per-seed outcome table. Only 's2' differs
# from the f"source_{key}" default: the queue key is 's2' but the table is
# source_semantic_scholar.
_OUTCOME_TABLE = {
    "arxiv": "source_arxiv",
    "openalex": "source_openalex",
    "s2": "source_semantic_scholar",
}


class SupabasePersistence:
    """Persistence backed by Supabase's PostgREST client."""

    def __init__(self, supabase: Client) -> None:
        self.supabase = supabase

    @classmethod
    def from_settings(cls, settings: SupabaseSettings) -> SupabasePersistence:
        if not (settings.supabase_url and settings.supabase_service_key):
            raise ValueError("supabase_url and supabase_service_key are required")
        return cls(create_client(settings.supabase_url, settings.supabase_service_key))

    def queue_visible_length(self, queue: str) -> int:
        """Messages available to read right now (not hidden by an in-flight VT).

        Wraps public.harvest_queue_visible_length (SECURITY DEFINER over
        pgmq.metrics) — pgmq.metrics itself is unreachable over PostgREST. The
        batch loop uses this to choose its fast path: when the count is already
        >= batch_size, read a full batch and flush in one shot; otherwise fall
        back to accumulate-with-timer. Treat it as a hint, not a guarantee —
        under multiple worker replicas it can be stale by the time we read.
        """
        resp = self.supabase.rpc("harvest_queue_visible_length", {"p_queue": queue}).execute()
        return int(resp.data or 0)  # type: ignore[arg-type]

    def read_batch(
        self,
        queue: str,
        *,
        n: int,
        vt_seconds: int,
        wait_seconds: float,
    ) -> list[dict[str, Any]]:
        """Accumulate up to ``n`` visible messages, then pgmq-read them.

        Waits up to ``wait_seconds`` for the queue to reach ``n`` visible
        messages (polling ``harvest_queue_visible_length``), then reads through
        ``pgmq_public.read`` with visibility timeout ``vt_seconds``. Returns raw
        pgmq rows (``msg_id``, ``message``, ``read_ct``, …) — not ``Seed``s;
        the worker resolves seeds via ``load_seeds``.
        """
        deadline_ns = time.time_ns() + int(wait_seconds * 1e9)
        while time.time_ns() < deadline_ns and self.queue_visible_length(queue) < n:
            time.sleep(0.1)

        q = self.supabase.schema("pgmq_public")
        resp = q.rpc(
            "read",
            {"queue_name": queue, "sleep_seconds": vt_seconds, "n": n},  # type: ignore[arg-type]
        ).execute()
        data = resp.data
        if not data or not isinstance(data, list):
            return []
        return [row for row in data if isinstance(row, dict)]

    def load_seed(self, paper_id: int) -> Seed:
        resp = (
            self.supabase.table("seeds")
            .select("paper_id, corpus, openalex_id, doi, arxiv_id, title")
            .eq("paper_id", paper_id)
            .single()
            .execute()
        )
        if not resp.data or not isinstance(resp.data, dict):
            raise ValueError(f"Seed not found for paper_id: {paper_id}")
        return Seed.model_validate(resp.data)

    def load_seeds(self, paper_ids: list[int]) -> dict[int, Seed]:
        if not paper_ids:
            return {}
        resp = (
            self.supabase.table("seeds")
            .select("paper_id, corpus, openalex_id, doi, arxiv_id, title")
            .in_("paper_id", paper_ids)
            .execute()
        )
        out: dict[int, Seed] = {}
        for row in resp.data or []:
            if not isinstance(row, dict):
                continue
            seed = Seed.model_validate(row)
            if seed.paper_id is None:
                continue
            out[seed.paper_id] = seed
        return out

    def save_seed(self, seed: Seed) -> int:
        # paper_id is `generated always as identity` — Postgres rejects an
        # explicit value (even NULL), so it must be excluded from the insert.
        resp = self.supabase.table("seeds").insert(seed.model_dump(exclude={"paper_id"})).execute()
        rows = resp.data or []
        if not rows or not isinstance(rows[0], dict) or "paper_id" not in rows[0]:
            raise ValueError(f"Failed to save seed: {seed}")
        return int(rows[0]["paper_id"])  # type: ignore[arg-type]

    def write(self, w: Write) -> None:
        for group in w.groups:
            if not group:
                continue
            row = group[0]
            (
                self.supabase.table(row.__table__)
                .upsert(
                    [r.model_dump(mode="json") for r in group],
                    on_conflict=",".join(row.__conflict__),
                )
                .execute()
            )

    def mark_not_found(self, source: str, paper_id: int) -> None:
        table = _OUTCOME_TABLE.get(source, f"source_{source}")
        self.supabase.table(table).upsert(
            {"paper_id": paper_id, "status": "not_found"},
            on_conflict="paper_id",
        ).execute()

    def mark_error(self, source: str, paper_id: int, error: str) -> None:
        table = _OUTCOME_TABLE.get(source, f"source_{source}")
        self.supabase.table(table).upsert(
            {"paper_id": paper_id, "status": "error", "error": error},
            on_conflict="paper_id",
        ).execute()
