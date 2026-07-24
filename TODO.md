# TODO: Additional harvest sources

Beyond the three sources already wired (`arxiv`, `s2` / Semantic Scholar, `openalex`), these are candidates for graph-layout theory and RAG/IR literature.

**What “adding a harvest source” needs (today):** free/usable API access, a client (official or community SDK, or thin `requests` wrapper), and an `identify` field on `public.seeds` (see `worker.SOURCES`). Existing seed columns: `arxiv_id`, `doi`, `openalex_id`, `title`.

Shared pattern for every source below (same as `arxiv` / `openalex` / `s2`):

1. `sources/<name>.py` — `PaperSource.get_papers(ids) -> dict[str, Record | None]`
2. `sources/models/<name>.py` — typed record + `from_dict` / SDK mapper
3. `persistence/models/<name>.py` — `source_<name>` row + `*_write(...) -> Write`
4. Supabase migration — `source_<name>`, `harvest_<name>` queue, dispatch filter
5. `worker.SOURCES["<name>"]` — `identify`, `build_write`, `batch_size`
6. Seed column if missing (migration + seed TOML / `NewSeed`)
7. Settings env (API email/key), version bump + CHANGELOG

Prefer DOI / arXiv id as join keys. Architecture: `docs/architecture/harvest.md`.

---

## ✅ Implemented in 0.13.0

The following 12 sources are now wired (source + models + persistence + migration +
`worker.SOURCES` + `queue_subscriptions` filter), and each is live-verified end-to-end
against a local Supabase stack via `v3/supabase/tests/local_harvest_loop.py`:

**unpaywall, crossref, dblp, openreview, acl_anthology, opencitations, datacite,
core, internet_archive, zenodo, figshare, osf** — plus the 4 seed identity columns
`dblp_id` / `openreview_id` / `anthology_id` / `ia_identifier`. See
`v3/harvest/CHANGELOG.md` (0.13.0) and the source table in `v3/harvest/README.md`.

Operational caveats live in that changelog: `opencitations` caps edges at
2000/direction (no server pagination), `core` free tier rate-limits (429 → redelivery),
`openreview` can hit an anti-bot 403 from non-allow-listed egress, `s2` is best-effort
on the shared pool.

**Deliberately NOT implemented (copyright):** Anna's Archive, Sci-Hub, Z-Library,
LibGen — an automated corpus-scale harvester against sources distributing works
without authorization is infringement. Legal OA coverage comes from unpaywall + core
+ internet_archive + zenodo/figshare/osf.

## Still deferred

- **bioRxiv / medRxiv** — only if the corpus expands biomedical (see below).
- **PubMed / Europe PMC, Scopus / Web of Science / Dimensions, Google Scholar** — the
  skip list below (biomedical skew, paywalled/overlapping, or no official API).

<details>
<summary>Original per-source implementation notes (kept for reference)</summary>

## Priority (metadata / venue / PDF)

### 1. Unpaywall — OA PDF rescue

| | |
|---|---|
| **Free access?** | Yes. `email=` required. ~100k/day. |
| **SDK** | Community: `unpywall`. Or REST. |
| **Seed field** | `doi` (existing). |
| **API** | `GET /v2/{doi}?email=` → `best_oa_location.url_for_pdf` |

**Implement**

```text
UnpaywallSource.get_papers(dois):
  for each doi: GET unpaywall; None if no OA location
  Record: doi, is_oa, best_pdf_url, host_type, license, updated
  identify: [seed.doi]
  batch_size: ~50–100 (serial OK; soft on 429)
  write: source_unpaywall (+ optional pdf_url column on a shared papers/pdf table later)
```

Settings: `UNPAYWALL_EMAIL` (required).

### 2. Crossref — authoritative DOI + references

| | |
|---|---|
| **Free access?** | Yes. Polite pool via `mailto=`. |
| **SDK** | Community: `habanero`. |
| **Seed field** | `doi` (existing). |
| **API** | `GET /works/{doi}` |

**Implement**

```text
CrossrefSource.get_papers(dois):
  habanero.Crossref(mailto=...).works(ids=doi)  # or batch via /works?filter=doi:...
  Record: doi, title, author, published, container-title, reference[], link[], abstract?
  identify: [seed.doi]
  batch_size: ~20–50 (polite pool: prefer list ≤3 concurrent)
  write: source_crossref; optional crossref_reference edge table from reference[]
```

Settings: `CROSSREF_MAILTO`.

### 3. DBLP — curated CS bibliography

| | |
|---|---|
| **Free access?** | Yes. No key. Be polite. |
| **SDK** | None official — `httpx`/`requests` to search + record endpoints. |
| **Seed field** | New `dblp_id` (DBLP key). Fallback: `doi` via search (weaker). |
| **API** | `/search/publ/api`, per-rec RDF/XML, SPARQL KG |

**Implement**

```text
migration: seeds.dblp_id text null
DblpSource.get_papers(keys):
  GET https://dblp.org/search/publ/api?q=... or fetch https://dblp.org/rec/{key}.xml
  Record: dblp_key, title, authors, venue, year, ee[], doi?
  identify: [seed.dblp_id]  # do not invent keys from fuzzy title in the worker
  batch_size: ~20–50
  write: source_dblp; backfill seeds.doi when ee/doi present
```

Discovery (seeding, not worker): SPARQL / venue dump → TOML seeds with `dblp_id`.

### 4. OpenReview — ML submissions + reviews

| | |
|---|---|
| **Free access?** | Yes for public notes. |
| **SDK** | Official: `openreview-py`. |
| **Seed field** | New `openreview_id` (forum id). |
| **API** | API v2 `api2.openreview.net` |

**Implement**

```text
migration: seeds.openreview_id text null
OpenReviewSource.get_papers(forum_ids):
  client = openreview.api.OpenReviewClient(baseurl=...)
  note = client.get_note(forum_id) / get_forum
  Record: id, title, abstract, authors, pdf url, venueid, replies (reviews) optional
  identify: [seed.openreview_id]
  batch_size: ~20–50
  write: source_openreview; optional openreview_review child rows
```

### 5. ACL Anthology — NLP/RAG full text

| | |
|---|---|
| **Free access?** | Yes (data repo + PDFs). |
| **SDK** | `acl-anthology` (load local/git data — not live batch HTTP). |
| **Seed field** | New `anthology_id` (e.g. `2023.acl-long.1`). |
| **API** | Anthology XML/data + `https://aclanthology.org/{id}.pdf` |

**Implement**

```text
migration: seeds.anthology_id text null
# Prefer sidecar index built at image build or first boot from anthology data
AnthologySource.get_papers(ids):
  lookup local Anthology index → metadata + pdf_url
  Record: anthology_id, title, authors, year, venue, doi?, pdf_url
  identify: [seed.anthology_id]
  batch_size: 100+ (local)
  write: source_acl_anthology
```

Worker stays offline-friendly; refresh anthology data on image rebuild or cron.

---

## Secondary

### OpenCitations (COCI)

| | |
|---|---|
| **Free access?** | Yes. |
| **SDK** | `opencitations_client`. |
| **Seed field** | `doi`. |
| **API** | `api.opencitations.net/index/v2` |

**Implement:** `identify=[doi]`; write citation edges (`citing_doi`, `cited_doi`) into `opencitations_edge` (or reuse graph tables). Batch small; use local dump backend if volume grows.

### DataCite

| | |
|---|---|
| **Free access?** | Public metadata read: yes. |
| **SDK** | `datacite` or REST. |
| **Seed field** | `doi`. |
| **API** | `api.datacite.org/dois/{doi}` |

**Implement:** same as Crossref but for dataset/software DOIs; `source_datacite` outcome; filter agency=DataCite when resolving ambiguous DOIs.

### CORE

| | |
|---|---|
| **Free access?** | Free tier **API key**. |
| **SDK** | Community / REST v3. |
| **Seed field** | `doi` (optional `core_id`). |
| **API** | `api.core.ac.uk/v3` |

**Implement:** settings `CORE_API_KEY`; `get_papers` by DOI; store OA full-text URL / download id in `source_core`.

### bioRxiv / medRxiv

| | |
|---|---|
| **Free access?** | Yes. |
| **SDK** | REST. |
| **Seed field** | `doi` or `biorxiv_id`. |

**Implement only if** corpus expands biomedical; otherwise leave unscheduled.

---

## Legitimate back channels (PDF / deposit rescue)

### Internet Archive

| | |
|---|---|
| **Free access?** | Yes for public items. Official Python lib; credentials only for restricted/upload. |
| **SDK** | Official: [`internetarchive`](https://internetarchive.readthedocs.io/) (`ia` CLI + `search_items` / `get_item`). |
| **Seed field** | New `ia_identifier` **or** resolve by `doi` / known URL via advanced search. Prefer storing `ia_identifier` once found. |
| **API** | Advanced Search + Metadata API + downloadable files on the item. |

**Implement**

```text
migration: seeds.ia_identifier text null
InternetArchiveSource.get_papers(ids):  # ia identifiers
  from internetarchive import get_item
  item = get_item(identifier)
  pick file: prefer application/pdf among item.files
  Record: identifier, title, date, pdf_name, pdf_url (archive.org/download/...), metadata
  identify: [seed.ia_identifier]
  # optional second path: search_items(f'doi:{doi}') only in a discovery/seeder job,
  # then persist ia_identifier — do not fuzzy-search inside the hot worker
  batch_size: ~10–20 (download metadata only in worker; binary PDF fetch is a later stage)
  write: source_internet_archive (status, ia_identifier, pdf_url, raw metadata jsonb)
```

Use IA for **dead OA links, author-deposited copies, old venue pages** — not as a substitute for Unpaywall when a DOI has a legal OA hit.

### Zenodo / Figshare / OSF (author deposits)

| | |
|---|---|
| **Free access?** | Yes (public records). |
| **SDK** | Zenodo REST; Figshare API; OSF API (community clients exist). |
| **Seed field** | Usually `doi` (DataCite-backed). Optional `zenodo_id` etc. |

**Implement:** thin sources keyed by DOI; store deposit PDF/file URL in `source_zenodo` / …; good companion to DataCite.

---

## Skip / low ROI (other)

| Source | Why |
|---|---|
| PubMed / Europe PMC | Biomedical skew for current corpora |
| Scopus / Web of Science / Dimensions | Paywalled; overlaps OpenAlex + S2 |
| Google Scholar | No official API |

---

## Suggested build order

1. Unpaywall (PDF rescue on existing DOIs)
2. Crossref (DOI truth + refs)
3. Internet Archive (legal last-resort PDF / snapshot)
4. DBLP (CS venue completeness + `dblp_id` seeding)
5. OpenReview + ACL Anthology (RAG/NLP full text)
6. OpenCitations / DataCite / CORE as needed

</details>
