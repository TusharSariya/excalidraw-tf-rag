"""Supabase implementation of the Persistence contract.

Source-agnostic by construction: ``write`` upserts whatever rows a mapper hands
it, reading each row's ``__table__``/``__conflict__``. It imports no source and
no row type — that absence is the seam working.
"""

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
    def from_settings(cls, settings: SupabaseSettings) -> "SupabasePersistence":
        if not (settings.supabase_url and settings.supabase_service_key):
            raise ValueError("supabase_url and supabase_service_key are required")
        return cls(create_client(settings.supabase_url, settings.supabase_service_key))

    def load_seed(self, paper_id: int) -> Seed:
        resp = (
            self.supabase.table("seeds")
            .select("paper_id, corpus, openalex_id, doi, arxiv_id, title")
            .eq("paper_id", paper_id)
            .single()
            .execute()
        )
        if not resp.data:
            raise ValueError(f"Seed not found for paper_id: {paper_id}")
        return Seed(**resp.data)

    def save_seed(self, seed: Seed) -> int:
        # paper_id is `generated always as identity` — Postgres rejects an
        # explicit value (even NULL), so it must be excluded from the insert.
        resp = (
            self.supabase.table("seeds")
            .insert(seed.model_dump(exclude={"paper_id"}))
            .single()
            .execute()
        )
        if not resp.data:
            raise ValueError(f"Failed to save seed: {seed}")
        return resp.data["paper_id"]

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
