from .acl_anthology import AclAnthologySource
from .arxiv import ArxivSource
from .base import PaperSource
from .core import CoreApiKeyMissing, CoreSource
from .crossref import CrossrefSource
from .datacite import DataCiteSource
from .dblp import DblpSource
from .figshare import FigshareSource
from .internet_archive import InternetArchiveSource
from .models.acl_anthology import AclAnthologyRecord
from .models.core import CoreWork
from .models.crossref import CrossrefWork
from .models.datacite import DataCiteRecord
from .models.dblp import DblpRecord
from .models.figshare import FigshareArticle
from .models.internet_archive import InternetArchiveItem
from .models.openalex import OpenAlexWork
from .models.opencitations import OpenCitationsRecord
from .models.openreview import OpenReviewNote
from .models.osf import OsfPreprint
from .models.s2 import S2Record
from .models.unpaywall import UnpaywallRecord
from .models.zenodo import ZenodoRecord
from .openalex import OpenAlexSource
from .opencitations import OpenCitationsSource
from .openreview import OpenReviewSource
from .osf import OsfSource
from .semantic_scholar import SemanticScholarSource
from .unpaywall import UnpaywallSource
from .zenodo import ZenodoSource

__all__ = [
    "PaperSource",
    "ArxivSource",
    "OpenAlexSource",
    "SemanticScholarSource",
    "UnpaywallSource",
    "CrossrefSource",
    "DblpSource",
    "OpenReviewSource",
    "AclAnthologySource",
    "OpenCitationsSource",
    "DataCiteSource",
    "CoreSource",
    "CoreApiKeyMissing",
    "InternetArchiveSource",
    "ZenodoSource",
    "FigshareSource",
    "OsfSource",
    "OpenAlexWork",
    "S2Record",
    "UnpaywallRecord",
    "CrossrefWork",
    "DblpRecord",
    "OpenReviewNote",
    "AclAnthologyRecord",
    "OpenCitationsRecord",
    "DataCiteRecord",
    "CoreWork",
    "InternetArchiveItem",
    "ZenodoRecord",
    "FigshareArticle",
    "OsfPreprint",
]
