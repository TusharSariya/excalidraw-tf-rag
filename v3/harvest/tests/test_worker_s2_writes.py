"""Worker persists list[Write] from S2 build_write."""

from __future__ import annotations

from harvest.persistence.base import Write
from harvest.persistence.models.s2 import SourceSemanticScholarRow, s2_writes
from harvest.sources.models.s2 import RelatedPaper, S2Record


def test_worker_write_loop_accepts_list_payload() -> None:
    """Mirrors worker.py: payload may be Write | list[Write]."""
    rec = S2Record(
        paper_id="s",
        title="T",
        citations=[RelatedPaper(paper_id="c1")],
        references=[],
    )
    payload = s2_writes(1, rec, chunk_size=1000)
    assert isinstance(payload, list)
    writes = payload if isinstance(payload, list) else [payload]
    assert all(isinstance(w, Write) for w in writes)
    assert isinstance(writes[-1].groups[0][0], SourceSemanticScholarRow)
