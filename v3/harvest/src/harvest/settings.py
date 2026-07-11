from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HarvestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        cli_parse_args=True,
        extra="ignore",
    )

    openalex_api_key: str | None = None
    arxiv_id: str | None = Field(default=None, description="Seed arXiv identifier")
    supabase_url: str | None = None
    supabase_key: str | None = None
