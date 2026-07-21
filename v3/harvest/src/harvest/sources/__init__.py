from .arxiv import ArxivSource
from .base import PaperSource
from .models.openalex import OpenAlexWork
from .models.s2 import S2Record
from .openalex import OpenAlexSource
from .semantic_scholar import SemanticScholarSource

__all__ = [
    "PaperSource",
    "ArxivSource",
    "OpenAlexSource",
    "SemanticScholarSource",
    "OpenAlexWork",
    "S2Record",
]
