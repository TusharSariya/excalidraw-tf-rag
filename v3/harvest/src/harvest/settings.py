"""Settings for the harvest entrypoints.

A shared base (`HarvestSettings`) resolves how to reach Postgres and any source
API keys from env / `.env` / CLI flags. Each entrypoint subclasses it with only
its own knobs, so `harvest-work` sees `--source` and `harvest-seed` sees
`--corpus`/`--arxiv_id`/`--doi` — neither exposes the other's flags.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class HarvestSettings(BaseSettings):
    """Shared config: how to reach Postgres + any source API keys."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        cli_parse_args=True,
        extra="ignore",
    )

    openalex_api_key: str | None = None  # OpenAlex polite pool (when OpenAlexSource lands)
    semantic_scholar_api_key: str | None = None  # optional — S2 also works unauthenticated


class SupabaseSettings(BaseSettings):
    """Settings for the Supabase client."""

    supabase_url: str | None = None
    supabase_service_key: str | None = None


class WorkerSettings(HarvestSettings):
    """`harvest-work`: drain ONE source's queue.

    `source` selects both the queue (`harvest_<source>`) and the handler.
    """

    source: str | None = None  # 'arxiv' | 'openalex' | 's2'
