import io
import logging
import os
import sys

import pdfplumber
import requests
from supabase import Client, create_client

from .settings import HarvestSettings
from .sources import ArxivSource, SemanticScholarSource

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("harvest")


def paper_exists(supabase: Client, arxiv_id: str) -> bool:
    logger.info("checking papers table for existing arxiv_id=%s", arxiv_id)
    existing = (
        supabase.table("papers")
        .select("paper_id")
        .eq("arxiv_id", arxiv_id)
        .execute()
    )
    if existing.data:
        logger.info("paper already in db, paper_id=%s", existing.data[0]["paper_id"])
        return True
    logger.info("paper not found in db, proceeding to harvest")
    return False


def download_and_extract_text(pdf_url: str) -> str:
    resp = requests.get(pdf_url)
    resp.raise_for_status()
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def main() -> None:
    logger.info("starting harvest")
    settings = HarvestSettings()
    logger.info("settings loaded: arxiv_id=%s", settings.arxiv_id)

    supabase = create_client(settings.supabase_url, settings.supabase_key)
    if paper_exists(supabase, settings.arxiv_id):
        return

    arxiv_source = ArxivSource()
    arxiv_paper = arxiv_source.get_paper(settings.arxiv_id)
    if arxiv_paper is None:
        return

    s2_source = SemanticScholarSource()
    authors = s2_source.get_authors(settings.arxiv_id)
    references = s2_source.get_references(settings.arxiv_id)
    citations = s2_source.get_citations(settings.arxiv_id)
    logger.info(
        "semantic scholar: authors=%d references=%d citations=%d",
        len(authors), len(references), len(citations),
    )

    open_access_url = s2_source.get_pdf_url(settings.arxiv_id)
    pdf_url = open_access_url or arxiv_paper.pdf_url
    text = download_and_extract_text(pdf_url)
    logger.info("extracted %d chars of text from %s", len(text), pdf_url)

    logger.info("harvest run complete for arxiv_id=%s", settings.arxiv_id)
