from rag_literature_rag.ingest.chunk import (
    chunk_pages,
    chunking_fingerprint,
    parse_markdown_blocks,
    strip_reference_sections,
)
from rag_literature_rag.ingest.extract import PageText
from rag_literature_rag.manifest import ManifestItem


def _item() -> ManifestItem:
    return ManifestItem(
        id="paper",
        title="Paper",
        source="curated",
        url="https://example.test/paper",
        status="ok",
        tags=["crossing"],
    )


def _filler_paragraphs(n: int, prefix: str = "Body paragraph") -> str:
    return "\n\n".join(
        f"{prefix} number {i} contains real content about the study." for i in range(n)
    )


def test_terminal_references_section_is_dropped():
    pages = [
        PageText(1, f"# Introduction\n\n{_filler_paragraphs(20)}"),
        PageText(
            2,
            "# References\n\n"
            "[1] A. Author. Some Paper Title. Venue, 2020.\n\n"
            "[2] B. Author. Another Paper. Venue, 2021.",
        ),
    ]
    chunks = chunk_pages(_item(), pages)
    combined = "\n".join(c.text for c in chunks)
    assert "Body paragraph number 0" in combined
    assert "Body paragraph number 19" in combined
    assert "[1] A. Author" not in combined
    assert "[2] B. Author" not in combined


def test_terminal_bibliography_and_acknowledgments_dropped():
    pages = [
        PageText(1, f"# Introduction\n\n{_filler_paragraphs(20)}"),
        PageText(
            2,
            "# Acknowledgments\n\nWe thank our funders for support.\n\n"
            "# Bibliography\n\n[1] A. Author. Some Paper Title. Venue, 2020.",
        ),
    ]
    chunks = chunk_pages(_item(), pages)
    combined = "\n".join(c.text for c in chunks)
    assert "Body paragraph number 0" in combined
    assert "We thank our funders" not in combined
    assert "[1] A. Author" not in combined


def test_midaocument_references_subsection_is_not_dropped():
    """A 'References' heading discussed early in the paper (not the terminal
    bibliography) must not nuke the rest of the document."""
    before = _filler_paragraphs(3, prefix="Early paragraph")
    after = _filler_paragraphs(10, prefix="Later paragraph")
    pages = [
        PageText(1, f"# Paper\n\n{before}"),
        PageText(
            2,
            "## References\n\nWe discuss references as a concept in citation graphs. "
            "This section explores how references function bibliographically.",
        ),
        PageText(3, after),
    ]
    chunks = chunk_pages(_item(), pages)
    combined = "\n".join(c.text for c in chunks)
    assert "Early paragraph number 0" in combined
    assert "citation graphs" in combined
    assert "Later paragraph number 9" in combined


def test_terminal_appendix_is_kept():
    pages = [
        PageText(1, f"# Introduction\n\n{_filler_paragraphs(20)}"),
        PageText(2, "# Appendix A\n\nSupplementary proofs and derivations live here."),
    ]
    chunks = chunk_pages(_item(), pages)
    combined = "\n".join(c.text for c in chunks)
    assert "Supplementary proofs" in combined


def test_opt_out_flag_preserves_old_behavior():
    pages = [
        PageText(1, f"# Introduction\n\n{_filler_paragraphs(20)}"),
        PageText(2, "# References\n\n[1] A. Author. Some Paper Title. Venue, 2020."),
    ]
    chunks = chunk_pages(_item(), pages, strip_references=False)
    combined = "\n".join(c.text for c in chunks)
    assert "[1] A. Author" in combined


def test_strip_reference_sections_reports_dropped_count():
    pages = [
        PageText(1, f"# Introduction\n\n{_filler_paragraphs(20)}"),
        PageText(2, "# References\n\n[1] A. Author. Some Paper Title. Venue, 2020."),
    ]
    blocks = parse_markdown_blocks(pages)
    kept, dropped = strip_reference_sections(blocks, doc_id="paper")
    assert dropped > 0
    assert len(kept) + dropped == len(blocks)


def test_fingerprint_changes_when_strip_references_flag_flips():
    on = chunking_fingerprint(None, strip_references=True)
    off = chunking_fingerprint(None, strip_references=False)
    assert on != off
    # Default matches the opt-in (on) behavior.
    assert chunking_fingerprint(None) == on
