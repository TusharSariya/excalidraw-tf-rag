import os

import arxiv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from semanticscholar import SemanticScholar


def main() -> None:
    print("Hello from harvest!")
    openalex_api_key = os.environ.get("OPENALEX_API_KEY")
    print(openalex_api_key)
    settings = HarvestSettings()
    print(settings.openalex_api_key)
    print(settings.arxiv_id)
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[settings.arxiv_id])))
    print(paper)
    print(paper.pdf_url)
    print(paper.source_url())

    sch = SemanticScholar()

    schpaper_id = "ARXIV:" + settings.arxiv_id
    authors = sch.get_paper_authors(schpaper_id, fields=["name", "authorId"])
    references = sch.get_paper_references(schpaper_id, ["title", "paperId"])
    citations = sch.get_paper_citations(schpaper_id, fields=["title", "paperId"], limit=100)

    print("authors:")
    for author in authors:
        print(author.name, author.authorId)

    print("refferences:")
    for ref in references:
        print(ref.paper.paperId, ref.paper.title)

    print("citations (first 100):")
    for cite in citations.items:
        print(cite.paper.paperId, cite.paper.title)


class HarvestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        cli_parse_args=True,
    )

    openalex_api_key: str | None = None
    arxiv_id: str | None = Field(default=None, description="Seed arXiv identifier")
