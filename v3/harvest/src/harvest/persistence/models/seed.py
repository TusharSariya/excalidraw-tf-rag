from pydantic import BaseModel, model_validator


class Seed(BaseModel):
    """Insert shape for public.seeds — paper_id is DB-generated."""

    corpus: str
    paper_id: int | None = None
    openalex_id: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    title: str | None = None
    dblp_id: str | None = None
    openreview_id: str | None = None
    anthology_id: str | None = None
    ia_identifier: str | None = None

    @model_validator(mode="after")
    def at_least_one_identity(self) -> "Seed":
        if not any(
            (
                self.openalex_id,
                self.doi,
                self.arxiv_id,
                self.title,
                self.dblp_id,
                self.openreview_id,
                self.anthology_id,
                self.ia_identifier,
            )
        ):
            raise ValueError(
                "NewSeed requires at least one of openalex_id, doi, arxiv_id, title, "
                "dblp_id, openreview_id, anthology_id, ia_identifier"
            )
        return self
