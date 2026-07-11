"""Settings for the harvest entrypoints.

A shared base (`HarvestSettings`) resolves how to reach Postgres and any source
API keys from env / `.env` / CLI flags. Each entrypoint subclasses it with only
its own knobs, so `harvest-work` sees `--source` and `harvest-seed` sees
`--corpus`/`--arxiv_id`/`--doi` — neither exposes the other's flags.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from supabase import Client, create_client


class HarvestSettings(BaseSettings):
    """Shared config: how to reach Postgres + any source API keys."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        cli_parse_args=True,
        extra="ignore",
    )

    # Supabase REST connection (supabase-py / PostgREST). The worker and seeder
    # talk to Postgres through the Supabase client, not a direct pg connection.
    # The service_role key is needed to reach the pgmq_public queue wrappers.
    supabase_url: str | None = None
    supabase_service_key: str | None = None

    # Source API keys — TBD which are actually required:
    openalex_api_key: str | None = None  # OpenAlex polite pool (when OpenAlexSource lands)
    semantic_scholar_api_key: str | None = None  # optional — S2 also works unauthenticated
    # arxiv needs no key.

    def client(self) -> "Client":
        """Build a Supabase client, or fail fast if the connection isn't configured."""
        if not (self.supabase_url and self.supabase_service_key):
            raise RuntimeError(
                "Supabase connection not configured: set SUPABASE_URL and "
                "SUPABASE_SERVICE_KEY (service_role key)"
            )
        return create_client(self.supabase_url, self.supabase_service_key)


class WorkerSettings(HarvestSettings):
    """`harvest-work`: drain ONE source's queue.

    `source` selects both the queue (`harvest_<source>`) and the handler.
    """

    source: str | None = None  # 'arxiv' | 'openalex' | 's2'


class SeederSettings(HarvestSettings):
    """`harvest-seed`: submit a paper into a corpus and fan it out to the queues."""

    corpus: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
