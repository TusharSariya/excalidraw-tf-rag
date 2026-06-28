# Graph-Layout-RAG: Metadata enrichment research (2026-06-28)

**Question:** what metadata can we pull from the papers (or external sources) to make retrieval perform better?

**Short answer:** the highest-leverage moves are *not* "stuff more metadata into the embedded chunk text" — that path is already A/B-proven NULL on this corpus (see §0). The wins are in **structured fields used as filters/routing**, **rank-time priors from citation data we already harvest but don't use in search**, and a **paper-level scientific embedding (SPECTER2) we can pull for free**. Plus the honest bottleneck: eval/gold, not retrieval.

---

## 0. The constraint that reframes everything

This corpus is **BM25-dominant over a lexically-rich vocabulary** (BM25 alone = 0.878/0.878 nDCG@10; tuned hybrid barely ties it). Two separate campaigns already tested *adding metadata as embedded chunk-text prefixes*:

- `section-v1` (Title/Year/Authors/Source/Section/Tags prefix) → **not promoted** (PDF +0.015 / catalog −0.010).
- `contextual-v1` (Anthropic contextual-retrieval prefix) → **fails the gate on both tracks** vs baseline *and* vs the section-v1 control.

Source: `docs/quality-campaign-2026-06-23.md`. Conclusion stated there: *"another additive indexed-text-prefix change that doesn't help a corpus where BM25 is already strong."*

**Implication:** do not expect a win from embedding more metadata strings into chunk text. Metadata helps here as (a) **filters / query routing**, (b) **score-level priors**, (c) a **second dense signal**, or (d) **better eval**. Every recommendation below is framed that way.

---

## 1. What we already have vs. what's actually used

Per-document manifest fields (`data/manifest.json`, 6,071 docs) and their coverage:

| Field | Coverage | In retrieval index? | In ranking? |
|---|---|---|---|
| title, source_url, tags, pipeline_categories | 100% | ✅ (embedded + filterable) | — |
| doi | 84% | alias_dois only | — |
| abstract | **60%** | ❌ **not indexed** | ❌ |
| authors | 49% | ✅ | ❌ |
| year | **38%** | ✅ + `year_min` filter | filter only |
| externalIds (S2/JGAA) | 14% | ❌ | ❌ |
| **venue / journal** | **0% (dropped)** | ❌ | ❌ |
| **concepts / topics / field-of-study** | **0% (never extracted)** | ❌ | ❌ |
| **citation counts** | see below | ❌ | relatedness only |

Two things already in the pipeline but **not used by search**:

1. **`data/citations.sqlite`** already holds `cited_by_count` **and** `influential_citation_count` for **53,136 papers** plus a **100,362-edge** citation graph. Today this powers only `query/citation_rank.py` (PPR + co-citation + a `w_prior=0.05` citation prior) for the *"cite related"* expansion path — **not** the main hybrid ranker.
2. The OpenAlex harvester (`harvest/openalex.py`) *uses* `concepts.id` as a search filter and `cited_by_count` as a sort key, and *reconstructs* abstracts from `abstract_inverted_index`, but **`primary_location.source` (venue) and `topics/concepts` are never written to the manifest** — they're read from the API response and discarded.

So a meaningful fraction of the "new metadata" is data we already touch and throw away.

---

## 2. External sources (all free, keyless-capable, high CS coverage)

Look up by DOI / arXiv ID / title. Add `?mailto=` (OpenAlex, Crossref) for the fast polite pool; a free S2 key gives 1 req/s.

### OpenAlex (`/works/doi:…`, `select=` to slim)
- **`primary_topic` + `topics[]`** — 4-level hierarchy domain→field→subfield→topic, each **scored**. ~85%+ classified; clean for CS. Top-1 topic accuracy ≈0.53 (0.72 with full metadata) → treat as a **soft** facet, not a hard gate. *Use `concepts` is deprecated — ignore it.*
- **`keywords[]`** — up to 5 scored phrases.
- **`primary_location.source.display_name` + `type`** — venue + journal/conference/preprint. Good filter facet.
- **`fwci`** — Field-Weighted Citation Impact; normalizes field+year bias → **better prior than raw `cited_by_count`**.
- **`referenced_works[]` / `related_works[]`** — feed the citation-graph / "more like this" path.
- **`abstract_inverted_index`** — free abstract backfill for the 40% of docs missing one.

### Semantic Scholar Graph API (`/paper/batch`, ≤500 IDs/req)
- **`tldr`** — AI one-sentence summary (SciTLDR). Short, query-shaped. Good CS coverage.
- **`embedding.specter_v2`** — precomputed **768-d SPECTER2** paper vector, **free, no GPU**.
- **`s2FieldsOfStudy`** — finer field labels with source provenance.
- **`influentialCitationCount`** — classifier-based "materially influenced" count; sharper prior than raw counts.
- **`citations.intents`/`contexts`** — citation intent (background/method/result) + sentence; method-intent edges are a strong relatedness signal.

### arXiv (`export.arxiv.org/api/query`) — for the arXiv slice (274+ docs, near-100% clean)
- **`primary_category`** + secondaries (`cs.DS`, `cs.CG`, `cs.HC`, `math.CO`) — author-assigned, the **cleanest free category label we can get**.
- **ACM class** (`F.2.2`…) and **MSC class** (`68R10`, `05Cxx`) when present.

### Crossref — lower priority
Authoritative venue/ISSN and reference DOIs when OpenAlex lacks them. `subject[]` is **sparse for CS conference proceedings** — don't rely on it.

### Controlled vocabulary
No official GD ontology. Best free composite label set = **arXiv primary_category + OpenAlex subfield/topic + MSC** (don't build a bespoke ontology). ACM CCS relevant nodes: *Human-centered computing → Visualization → Graph drawings*; *Theory of computation → Computational geometry*; *Mathematics of computing → Graph algorithms*.

---

## 3. Does each signal actually help retrieval? (evidence)

- **(a) Abstract / TLDR as chunk context — supported in general, but gate it here.** Metadata-prefix RAG papers report large QA gains (*Utilizing Metadata for Better RAG* arXiv:2601.11863; metadata-RAG for finance arXiv:2510.24402, ~30% LLM-judged). **But** our own `section-v1`/`contextual-v1` arms already failed. TLDR is the *one* prefix worth a re-test because it's short and query-shaped (unlike the section/context prefixes that failed) — prior is it'll also be NULL; gate it, don't ship blind.
- **(b) Topic / field-of-study / venue — supported, mainly as FILTERS / routing**, not embedded text (*SRAG* arXiv:2603.26670). Matches our corpus character.
- **(c) Citation / recency priors — supported but SECONDARY**, as small rank-time blends, not retrieval signals (LtR scientific search arXiv:1611.01400; WSDM'10 recency). Use `fwci` / `influentialCitationCount` + mild recency, **small weight** — tie-breaking, not match-fixing. Consistent with our own finding that citation features dominate *relatedness* but are weak for text retrieval.
- **(d) SPECTER2 — supported for paper-level tasks.** Beats general embedders on SciRepEval; but it's a **title+abstract** embedder, not a passage embedder — test as a **paper-level candidate generator / reranking feature**, not a replacement for the tuned Qwen-4B chunk vectors. Hybrid still wins (*Sparse Meets Dense* arXiv:2401.04055; *DORIS-MAE* arXiv:2310.04678).
- **(e) Metadata query routing — supported.** Route a query that names a category ("orthogonal", "cs.HC user study") to a facet filter before hybrid.

---

## 4. Prioritized recommendations (highest ROI first)

1. **Wire the citation prior we already have into the main hybrid ranker (not just relatedness).** `citations.sqlite` already has `cited_by_count` + `influential_citation_count` for 53k papers. Add a small `fwci`-style / log-influential prior as a rank-time blend behind a flag, gate it. **Zero new harvesting.** *(Evidence (c); honest expectation: small, tie-breaking.)*
2. **Add structured filter facets: `venue`, `arxiv_primary_category`, `openalex_subfield`.** Persist them on `ManifestItem` + index columns (mirror the existing `year`/`tag`/`source` filter machinery in `query/retrieve.py`). Enables routing/scoping. Venue + arXiv category are dropped/never-extracted today but already in the API responses we fetch. *(Evidence (b),(e).)*
3. **Backfill the 40% missing abstracts + 62% missing years + 51% missing authors** from OpenAlex `abstract_inverted_index` / `publication_year` / `authorships`. Improves filter coverage and any abstract-based feature; corrects the `year_min` filter's silent under-coverage. *(One batched OpenAlex pass over the manifest.)*
4. **Pull free SPECTER2 vectors (`embedding.specter_v2`) from S2 `/paper/batch` and A/B them as a paper-level reranker / candidate generator** against the current pipeline. Free, no GPU. *(Evidence (d).)*
5. **Re-test ONLY the TLDR prefix** as a chunk-context line (against the section-v1 control, same gate). Expect NULL given §0; cheap to falsify. *(Evidence (a).)*
6. **Strengthen "cite related"** with `referenced_works`/`related_works` + S2 citation **intents** (method-intent edges). Extends the existing `citation_rank.py` path. *(Evidence (c).)*

**Skip:** OpenAlex `concepts` (deprecated → use `topics`), `mesh`/SDG (≈0% coverage), Crossref `subject` (sparse for CS), bespoke GD ontology.

---

## 5. The bigger lever (caveat to the whole question)

Per the campaign record and prior findings, **retrieval here is already well-tuned; the measured bottleneck was sparse gold labels**, not ranking. Metadata's biggest payoff may be in **eval**: use OpenAlex `subfield` / arXiv category to *stratify and expand the gold set* (coverage per topic), and to detect topical holes. A retrieval tweak that can't be measured won't be promotable — invest in the gold set in parallel with §4.

---

## 6. Fields to add for the *research-tool* job (not ranking)

Different goal: not nDCG, but helping a human/agent **triage → trust → navigate → cite → reproduce**. Today a result row surfaces title, score, page, source_url, year, authors, evidence snippet. The fields below add research affordances. All free from the §2 sources unless noted.

### Triage — "is this worth reading?"
- **`tldr`** (S2) — one-sentence summary shown inline next to each hit. Biggest single triage win.
- **abstract** — backfill the 40% missing (OpenAlex `abstract_inverted_index`), surface on expand.
- **`documentKind` / publicationTypes** — today **every doc is tagged `"paper"`**. Distinguishing **survey / handbook-chapter / thesis / tool-doc / primary-research** is high-value for research (a survey is a different read than a 4-page GD poster). S2 `publicationTypes` (`Review`, `JournalArticle`, `Conference`), OpenAlex `type`.
- **citation count + "seminal vs recent"** badge — `cited_by_count` (S2/OpenAlex) and `counts_by_year` (citation trajectory: rising vs dormant).

### Trust / assess — "can I rely on it?"
- **`is_retracted` / retraction flag** (OpenAlex `is_retracted`, Crossref update-to) — **critical integrity field for any research tool**; surface a warning, never silently cite a retracted paper.
- **peer-reviewed vs preprint** — OpenAlex `type` (`preprint`) / `primary_location.version`; arXiv-only vs published DOI.
- **venue + venue tier** — `primary_location.source.display_name`; GD/JGAA/TVCG vs unranked.
- **`fwci`** (OpenAlex) — field-weighted impact percentile: is it actually influential *for its field and age*, not just old.
- **version** — arXiv v1 vs latest vs the published DOI (avoid quoting a superseded preprint).

### Navigate / discover — "what's connected?"
- **in-corpus citation cross-refs** — *"N other papers in your corpus cite this / are cited by this."* You already have the 100k-edge `citations.sqlite` + `in_corpus` flag — this is mostly a **surfacing** task, very powerful for a local literature tool.
- **references[] + cited_by[]** with titles (OpenAlex `referenced_works`, S2 `/references`,`/citations`) — jump up/down the citation tree from a hit.
- **`related_works`** (OpenAlex) — "more like this" without a new query.
- **citation intents** (S2 `intents`: background/method/result) — "who *builds on* this method" vs merely name-checks it.
- **author → other corpus papers** — `authorship` table already exists; surface "3 more by North in corpus."
- **topic / arXiv-category facets** — browse the corpus by subfield, not just search it.

### Cite / export — "put it in my writeup"
- **BibTeX / formatted citation** — free via Crossref content negotiation (`Accept: application/x-bibtex` on the DOI) or DataCite. A one-keystroke "cite this" is a defining research-tool feature.
- **full biblio** — volume/issue/pages/full author list (OpenAlex `biblio`, Crossref) — needed for a complete citation; we currently truncate authors.
- **stable IDs** — DOI, S2 `CorpusId`, OpenAlex ID, **DBLP key** (S2 `externalIds.DBLP`) for cross-linking.

### Reproduce / go deeper
- **best OA PDF + license** — OpenAlex `best_oa_location` / `open_access.oa_url`, S2 `openAccessPdf`. A guaranteed-readable link beats a paywalled DOI.
- **code / dataset links** — Papers-with-Code association (free API, keyed by arXiv ID) — strong for the algorithmic GD slice.
- **figure / table captions** — you already run docling; surfacing extracted figure captions + equation blocks makes deep-reads faster (no new source needed).

### Top picks if you add only a few
1. **`tldr`** inline (triage). 2. **`is_retracted`** + preprint/published flag (trust). 3. **in-corpus citation cross-refs** from data you already have (navigate). 4. **BibTeX export** (cite). 5. **survey/doc-kind** classification (triage). None of these need to touch the ranker — they're surfacing + a couple of batched backfills.

### Sources
OpenAlex Work object & Topics docs; S2 Graph API docs & release notes; Crossref REST API tips; arXiv API manual & CS subject classes; SPECTER2 (Ai2); arXiv:2601.11863, 2510.24402, 2603.26670, 2401.04055, 2310.04678, 1611.01400; WSDM'10 recency ranking. Internal: `docs/quality-campaign-2026-06-23.md`, `query/citation_rank.py`, `harvest/openalex.py`, memory notes (search well-tuned / gold-label bottleneck; citation features dominate relatedness, weak on text retrieval).

---

## Implementation outcomes (2026-06-28)

Shipped as three commits/PRs, mapping to the prioritized recommendations above:

- **P1 — surfacing.** `harvest enrich-metadata` (OpenAlex `/works/doi:` + title/arXiv fallback, S2 `/paper/batch`, arXiv API) → `papers_meta` + `doc_specter2` tables in `data/citations.sqlite` and filter fields on `data/manifest.json`. Query results now surface `tldr`, `fwci`, `venue`, `genre`, `is_retracted` (⚠), `cited_by_count`, `in_corpus_cited_by_count`, `oa_pdf_url`, `bibtex`; new `cite bibtex <doc_id>` command. Joins at query time — no rebuild needed; degrades gracefully when a doc isn't enriched.
- **P2 — filter columns (code).** Filter-only manifest/index columns `venue`, `arxiv_category`, `genre`, `venue_type`, `oa_version`, `is_retracted` as real pre-filters (LanceDB `WHERE prefilter=True` dense; indexed Tantivy fields sparse), with `--venue` / `--arxiv-category` / `--genre` / `--exclude-retracted` query flags. NOT embedded (embedding-metadata is A/B-NULL here, §0). **Live-index rebuild pending** — the schema gains columns, so `ingest --force --rebuild` is required before the filters bite.
- **P3 — citation prior (default OFF).** `--citation-prior-weight FLOAT` rank-time blend, default `0.0`, eval-gated.

### OpenAlex coverage

**82% of items enriched (4936/6050)**, and **98% where a real DOI exists**, after fixing a DOI-lookup-URL bug (`/works/https://doi.org/<doi>` → `/works/doi:<doi>`) that had silently capped DOI items at ~4%. The remaining **~1114 no-DOI tail** (mostly JGAA) is blocked on OpenAlex's `/works?search=` daily budget and converges via a later resume pass (re-run `enrich-metadata` without `--force`).

### Citation-graph keying limitation (predicts P3 NULL)

Only **~22% of chunk doc_ids map into the CitationGraph's `doc_to_oa`** — the graph keys on `s2-`/`openalex-` ids while chunks key on `doi-`/`arxiv-` ids. So `in_corpus_cited_by_count` and the citation prior are **mostly neutral**, predicting the P3 ranking gate will be NULL. The prior ships **OFF** accordingly; closing the keying gap is prerequisite work before the gate is worth running.

### Validation

End-to-end validation passed on the **desktop NL/4B path**: `fwci`, `venue`, `tldr`, `bibtex`, and cited-by counts surfaced on live queries. `citations.sqlite` (incl. `papers_meta`) now syncs via `gpu_sync_to_remote.sh` (WAL-checkpointed; `RAG_SYNC_CITATIONS=0` to skip). The `query_remote.sh` chunk-count guard no longer hardcodes `41083` (it changes after a schema rebuild) — pin `GRAPH_RAG_NL_EXPECTED_CHUNKS` for exact matching, else a relaxed floor warns-not-fails.
</content>
</invoke>
