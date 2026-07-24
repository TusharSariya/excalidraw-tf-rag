import gzip
import logging
import re

import requests

from .base import PaperSource
from .models.acl_anthology import AclAnthologyRecord

logger = logging.getLogger("harvest.acl_anthology")

# The native batch surface: the full-corpus dump — one gzipped file containing every
# ACL Anthology record. Downloaded ONCE, parsed into an in-memory index keyed by
# anthology_id, after which every lookup is a local dict hit. This is what makes the
# source an offline-friendly local index with trivial batch lookups.
_CORPUS_URL = "https://aclanthology.org/anthology.bib.gz"

# The full corpus is a large download; allow it a generous timeout. A slow/failed
# download is treated as transient and raised so pgmq redelivers the batch — the
# index is only fetched once per source instance, so this cost is amortised across
# every batch the worker drains, not paid per message.
_TIMEOUT_SECONDS = 300.0

# @<type>{  — the start of a BibTeX entry. @comment/@string/@preamble are skipped.
_ENTRY_START = re.compile(r"@(?P<type>\w+)\s*\{", re.IGNORECASE)
_SKIP_TYPES = frozenset({"comment", "string", "preamble"})


def _anthology_id_from_url(url: str | None) -> str | None:
    """``https://aclanthology.org/N19-1423/`` -> ``N19-1423``.

    The anthology_id is the request key, not a standalone bib field — it is the last
    path segment of the record ``url``.
    """
    if not url:
        return None
    return url.rstrip("/").rsplit("/", 1)[-1] or None


def _parse_fields(body: str) -> tuple[str, dict[str, str]]:
    """Parse a BibTeX entry body into ``(bibkey, {field: value})``.

    Handles ``key = "value"``, ``key = {value}``, and bare values (``month = jun``),
    with brace-aware scanning so commas and quotes nested inside a value do not end
    it early. Values are returned verbatim (braces intact) — the model strips TeX
    braces at typing time.
    """
    comma = body.find(",")
    if comma == -1:
        return body.strip(), {}
    bibkey = body[:comma].strip()
    rest = body[comma + 1 :]
    fields: dict[str, str] = {}
    i, n = 0, len(rest)
    while i < n:
        while i < n and rest[i] in " \t\r\n,":
            i += 1
        eq = rest.find("=", i)
        if eq == -1:
            break
        name = rest[i:eq].strip().lower()
        i = eq + 1
        while i < n and rest[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        c = rest[i]
        if c == '"':
            i += 1
            depth, buf = 0, []
            while i < n:
                ch = rest[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                elif ch == '"' and depth == 0:
                    i += 1
                    break
                buf.append(ch)
                i += 1
            value = "".join(buf)
        elif c == "{":
            depth, i = 1, i + 1
            buf = []
            while i < n and depth > 0:
                ch = rest[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                if depth > 0:
                    buf.append(ch)
                i += 1
            value = "".join(buf)
        else:
            buf = []
            while i < n and rest[i] not in ",\n":
                buf.append(rest[i])
                i += 1
            value = "".join(buf).strip()
        if name:
            fields[name] = value
    return bibkey, fields


def parse_bibtex_entries(text: str) -> list[dict]:
    """Split a BibTeX document into parsed entry dicts keyed for indexing.

    Each entry becomes ``{anthology_id, entry_type, bibkey, bibtex: {…}, raw_bibtex}``
    — the shape ``AclAnthologyRecord.from_dict`` types over. Entries whose ``url``
    yields no anthology_id are still returned (indexed under None gets dropped later).
    """
    entries: list[dict] = []
    n = len(text)
    for m in _ENTRY_START.finditer(text):
        entry_type = m.group("type").lower()
        if entry_type in _SKIP_TYPES:
            continue
        start = m.end()
        depth, j = 1, start
        while j < n and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        body = text[start : j - 1]
        raw_bibtex = text[m.start() : j]
        bibkey, fields = _parse_fields(body)
        entries.append(
            {
                "anthology_id": _anthology_id_from_url(fields.get("url")),
                "entry_type": entry_type,
                "bibkey": bibkey,
                "bibtex": fields,
                "raw_bibtex": raw_bibtex,
            }
        )
    return entries


def build_index(entries: list[dict]) -> dict[str, dict]:
    """Index parsed entries by anthology_id; first entry wins, keyless dropped."""
    index: dict[str, dict] = {}
    for entry in entries:
        aid = entry.get("anthology_id")
        if aid:
            index.setdefault(aid, entry)
    return index


class AclAnthologySource(PaperSource):
    """PaperSource implementation backed by the ACL Anthology BibTeX corpus.

    ``identifier`` is an anthology_id (``N19-1423``, ``2020.emnlp-main.550``) — the
    ACL Anthology's own paper id, carried on ``seeds.anthology_id`` and gated by the
    ``{"any_of": ["anthology_id"]}`` dispatch filter, so seeds without one never
    reach this source. No API key is required and none is used (probed live, no auth).

    **Batch via the full-corpus dump.** ACL Anthology has no multi-id JSON endpoint;
    its native batch surface is the whole-corpus gzip ``anthology.bib.gz``. This
    source downloads and parses it ONCE (cached on the instance), then resolves every
    requested id against the in-memory index — one HTTP round-trip serves an entire
    batch and every subsequent batch the worker drains. That is the mandatory
    batch path, not a per-id ``.bib`` loop. Because lookups are local, a large
    ``batch_size`` (e.g. 500) is appropriate for the worker SOURCES entry.

    A requested id absent from the corpus is terminal -> ``None`` (the worker marks
    it not_found). A transient failure downloading the corpus (429 / 5xx / timeout)
    raises so the worker leaves the messages unarchived and pgmq redelivers after the
    visibility timeout — no in-process backoff, which would risk overrunning
    ``worker.VT_SECONDS``.

    Identity is never resolved by fuzzy title search: give it an anthology_id or it
    returns None.
    """

    def __init__(self, corpus_url: str = _CORPUS_URL) -> None:
        self._corpus_url = corpus_url
        self._session = requests.Session()
        # Lazy, cached per-instance: the corpus is fetched+parsed on first use and
        # reused for every later batch. None until the first successful download.
        self._index: dict[str, dict] | None = None

    def get_papers(self, identifiers: list[str]) -> dict[str, AclAnthologyRecord | None]:
        """Resolve anthology_ids against the corpus index; misses map to None.

        Missing ids are terminal (don't retry). A failed corpus download raises so
        the worker leaves the messages unarchived for pgmq redelivery; retrying
        in-process instead would risk blocking past ``worker.VT_SECONDS``, which lets
        pgmq hand the same batch to a second worker while this one is still sleeping.
        """
        out: dict[str, AclAnthologyRecord | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching metadata for %s", identifiers)
        index = self._ensure_index()
        for req in identifiers:
            entry = index.get(req.strip())
            if entry is None:
                logger.warning("no result for %s (not found)", req)
                continue
            out[req] = AclAnthologyRecord.from_dict(entry)
        return out

    def _ensure_index(self) -> dict[str, dict]:
        """Download + parse the corpus dump once; cache and reuse the index.

        Transient HTTP (429 / 5xx) and any non-200 raise so pgmq redelivers.
        """
        if self._index is not None:
            return self._index

        logger.info("downloading ACL Anthology corpus from %s", self._corpus_url)
        resp = self._session.get(self._corpus_url, timeout=_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            # 429 (rate limit) and 5xx are transient — raise so pgmq redelivers.
            logger.error("corpus fetch failed (HTTP %s)", resp.status_code)
            resp.raise_for_status()
            # Non-200, non-error status (e.g. an unexpected 3xx) — treat as transient
            # rather than silently index nothing and mark every id not_found.
            raise RuntimeError(f"unexpected HTTP {resp.status_code} fetching ACL corpus")

        text = gzip.decompress(resp.content).decode("utf-8", errors="replace")
        self._index = build_index(parse_bibtex_entries(text))
        logger.info("indexed %d ACL Anthology records", len(self._index))
        return self._index
