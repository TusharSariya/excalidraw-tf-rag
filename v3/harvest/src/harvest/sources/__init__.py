from .arxiv import ArxivSource
from .base import Author, Paper, PaperSource
from .models.s2 import S2Record
from .openalex import OpenAlexSource
from .semantic_scholar import SemanticScholarSource

__all__ = [
    "Author",
    "Paper",
    "PaperSource",
    "ArxivSource",
    "OpenAlexSource",
    "SemanticScholarSource",
    "S2Record",
]
