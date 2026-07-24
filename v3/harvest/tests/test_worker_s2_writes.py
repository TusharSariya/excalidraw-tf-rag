"""Worker streams S2 writes via iter_s2_writes + related chunks."""

from __future__ import annotations

from harvest.persistence.base import Write
from harvest.persistence.models.s2 import SourceSemanticScholarRow, iter_s2_writes
from harvest.sources.models.s2 import RelatedPaper, S2Record


def test_iter_s2_writes_with_live_chunks() -> None:
    rec = S2Record(paper_id="s", title="T", citation_count=2, reference_count=1)
    cites = iter([[RelatedPaper(paper_id="c1")], [RelatedPaper(paper_id="c2")]])
    refs = iter([[RelatedPaper(paper_id="r1")]])
    writes = list(iter_s2_writes(1, rec, citation_chunks=cites, reference_chunks=refs))
    assert all(isinstance(w, Write) for w in writes)
    assert len(writes) == 1 + 2 + 1 + 1  # subject + 2 cite + 1 ref + outcome
    assert isinstance(writes[-1].groups[0][0], SourceSemanticScholarRow)
